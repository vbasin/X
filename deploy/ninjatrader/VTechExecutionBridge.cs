// NinjaTrader 8 Strategy: atomic file bridge for VTech execution commands.
// Install manually under Documents\NinjaTrader 8\bin\Custom\Strategies and compile.
// Safe defaults: Sim101 only, live orders disabled, one position, native stop/target OCO.

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using System.Threading;
using Newtonsoft.Json.Linq;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class VTechExecutionBridge : Strategy
    {
        private readonly object ioLock = new object();
        private readonly Dictionary<string, string> commandBySignal = new Dictionary<string, string>();
        private readonly Dictionary<string, JObject> commands = new Dictionary<string, JObject>();
        private Timer pollTimer;
        private int pollQueued;
        private DateTime lastHeartbeat = DateTime.MinValue;
        private bool killFlattenSubmitted;
        private string activeSignalName;
        private string activeCommandId;
        private DateTime entryFilledAt = DateTime.MinValue;
        private bool timeoutFlattenSubmitted;

        [NinjaScriptProperty]
        [Display(Name = "Bridge directory", Order = 1, GroupName = "VTech")]
        public string BridgeDirectory { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Allow LIVE commands", Order = 2, GroupName = "VTech")]
        public bool AllowLiveOrders { get; set; }

        [NinjaScriptProperty]
        [Range(100, 5000)]
        [Display(Name = "Poll milliseconds", Order = 3, GroupName = "VTech")]
        public int PollMilliseconds { get; set; }

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "VTechExecutionBridge";
                Description = "Consumes risk-approved VTech commands and returns broker events.";
                Calculate = Calculate.OnEachTick;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.UniqueEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                StartBehavior = StartBehavior.WaitUntilFlat;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                BridgeDirectory = @"C:\VTechExecution";
                AllowLiveOrders = false;
                PollMilliseconds = 250;
            }
            else if (State == State.DataLoaded)
            {
                Directory.CreateDirectory(Path.Combine(BridgeDirectory, "outbox"));
                Directory.CreateDirectory(Path.Combine(BridgeDirectory, "inbox"));
                Directory.CreateDirectory(Path.Combine(BridgeDirectory, "processed"));
                pollTimer = new Timer(PollTimer, null, PollMilliseconds, PollMilliseconds);
            }
            else if (State == State.Terminated)
            {
                if (pollTimer != null)
                    pollTimer.Dispose();
            }
        }

        protected override void OnBarUpdate()
        {
            // Orders are dispatched by TriggerCustomEvent from the timer, not by bar cadence.
        }

        private void PollTimer(object state)
        {
            if (Interlocked.Exchange(ref pollQueued, 1) != 0)
                return;
            try
            {
                TriggerCustomEvent(o =>
                {
                    try { ProcessCommands(); }
                    finally { Interlocked.Exchange(ref pollQueued, 0); }
                }, null);
            }
            catch
            {
                Interlocked.Exchange(ref pollQueued, 0);
            }
        }

        private void ProcessCommands()
        {
            EmitHeartbeat();
            if (State != State.Realtime)
                return;
            bool killed = File.Exists(Path.Combine(BridgeDirectory, "KILL_SWITCH"));
            if (killed && Position.MarketPosition != MarketPosition.Flat && !killFlattenSubmitted)
            {
                killFlattenSubmitted = true;
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong("VTKill", activeSignalName);
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort("VTKill", activeSignalName);
            }
            if (!killed && Position.MarketPosition == MarketPosition.Flat)
                killFlattenSubmitted = false;
            if (Position.MarketPosition != MarketPosition.Flat && !timeoutFlattenSubmitted
                && !String.IsNullOrEmpty(activeCommandId) && entryFilledAt != DateTime.MinValue)
            {
                int maxHoldingSeconds = (int)commands[activeCommandId]["max_holding_seconds"];
                if ((DateTime.UtcNow - entryFilledAt).TotalSeconds >= maxHoldingSeconds)
                {
                    timeoutFlattenSubmitted = true;
                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong("VTTimeout", activeSignalName);
                    else if (Position.MarketPosition == MarketPosition.Short)
                        ExitShort("VTTimeout", activeSignalName);
                }
            }
            if (Position.MarketPosition == MarketPosition.Flat)
            {
                entryFilledAt = DateTime.MinValue;
                timeoutFlattenSubmitted = false;
            }
            foreach (string path in Directory.GetFiles(Path.Combine(BridgeDirectory, "outbox"), "*.json").OrderBy(p => p))
            {
                JObject cmd;
                try { cmd = JObject.Parse(File.ReadAllText(path)); }
                catch { continue; }

                string commandId = (string)cmd["command_id"];
                if (String.IsNullOrWhiteSpace(commandId))
                    continue;
                string marker = Path.Combine(BridgeDirectory, "processed", commandId + ".done");
                if (File.Exists(marker))
                    continue;

                string reason = ValidateCommand(cmd);
                // Marker precedes submission: after a crash the command cannot be duplicated.
                File.WriteAllText(marker, DateTime.UtcNow.ToString("o"));
                Emit(cmd, reason == null ? "COMMAND_RECEIVED" : "ORDER_REJECTED", null, 0, 0, reason);
                if (reason != null)
                    continue;

                string signalName = "VT" + commandId.Replace("cmd_", "").Substring(0, 16);
                activeSignalName = signalName;
                activeCommandId = commandId;
                commandBySignal[signalName] = commandId;
                commands[commandId] = cmd;
                int quantity = (int)cmd["quantity"];
                double targetTicks = (double)cmd["target_points"] / TickSize;
                double stopTicks = (double)cmd["stop_points"] / TickSize;
                SetProfitTarget(signalName, CalculationMode.Ticks, targetTicks);
                SetStopLoss(signalName, CalculationMode.Ticks, stopTicks, false);

                bool buy = String.Equals((string)cmd["action"], "BUY", StringComparison.OrdinalIgnoreCase);
                bool limit = String.Equals((string)cmd["entry_order_type"], "LIMIT", StringComparison.OrdinalIgnoreCase);
                if (limit)
                {
                    double limitPrice = (double)cmd["limit_price"];
                    if (buy) EnterLongLimit(quantity, limitPrice, signalName);
                    else EnterShortLimit(quantity, limitPrice, signalName);
                }
                else
                {
                    if (buy) EnterLong(quantity, signalName);
                    else EnterShort(quantity, signalName);
                }
                Emit(cmd, "ORDER_SUBMITTED", "ENTRY", 0, 0, null);
            }
        }

        private string ValidateCommand(JObject cmd)
        {
            if (File.Exists(Path.Combine(BridgeDirectory, "KILL_SWITCH"))) return "kill_switch_active";
            if ((string)cmd["schema_version"] != "v1") return "unsupported_schema";
            DateTime expires;
            if (!DateTime.TryParse((string)cmd["expires_at"], out expires) || expires.ToUniversalTime() < DateTime.UtcNow) return "expired_command";
            if (!String.Equals((string)cmd["account"], Account.Name, StringComparison.OrdinalIgnoreCase)) return "account_mismatch";
            if (!String.Equals((string)cmd["instrument"], Instrument.FullName, StringComparison.OrdinalIgnoreCase)) return "instrument_mismatch";
            if (!String.Equals((string)cmd["execution_style"], "OCO", StringComparison.OrdinalIgnoreCase)) return "oco_required";
            if ((int?)cmd["quantity"] == null || (int)cmd["quantity"] < 1) return "invalid_quantity";
            if (Position.MarketPosition != MarketPosition.Flat) return "strategy_position_not_flat";
            if (Account.Positions.Any(p => p.Instrument.FullName == Instrument.FullName
                && p.MarketPosition != MarketPosition.Flat)) return "account_instrument_not_flat";
            string mode = ((string)cmd["mode"] ?? "").ToUpperInvariant();
            bool simAccount = Account.Name.StartsWith("Sim", StringComparison.OrdinalIgnoreCase);
            if (mode == "SIM" && !simAccount) return "sim_command_requires_sim_account";
            if (mode == "LIVE" && (!AllowLiveOrders || simAccount)) return "live_orders_not_authorized";
            if (mode != "SIM" && mode != "LIVE") return "invalid_mode";
            return null;
        }

        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice,
            int quantity, int filled, double averageFillPrice, OrderState orderState,
            DateTime time, ErrorCode error, string nativeError)
        {
            string signalName = order.FromEntrySignal;
            if (String.IsNullOrEmpty(signalName) && commandBySignal.ContainsKey(order.Name))
                signalName = order.Name;
            if (String.IsNullOrEmpty(signalName) || !commandBySignal.ContainsKey(signalName))
                return;
            string commandId = commandBySignal[signalName];
            JObject cmd = commands[commandId];
            string role = Role(order, signalName);
            if (orderState == OrderState.Accepted || orderState == OrderState.Working)
                Emit(cmd, "ORDER_ACCEPTED", role, 0, 0, null, order.OrderId, orderState.ToString());
            else if (orderState == OrderState.Rejected)
                Emit(cmd, "ORDER_REJECTED", role, 0, 0, error + ": " + nativeError, order.OrderId, orderState.ToString());
            else if (orderState == OrderState.Cancelled && role == "ENTRY")
                Emit(cmd, "ENTRY_CANCELLED", role, 0, 0, nativeError, order.OrderId, orderState.ToString());
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId,
            double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            Order order = execution.Order;
            string signalName = order == null ? null : order.FromEntrySignal;
            if (String.IsNullOrEmpty(signalName) && order != null && commandBySignal.ContainsKey(order.Name))
                signalName = order.Name;
            if (String.IsNullOrEmpty(signalName) || !commandBySignal.ContainsKey(signalName))
                return;
            string commandId = commandBySignal[signalName];
            JObject cmd = commands[commandId];
            string role = Role(order, signalName);
            string eventType = order.OrderState == OrderState.Filled ? "FILLED" : "PARTIAL_FILL";
            if (role == "ENTRY" && entryFilledAt == DateTime.MinValue)
                entryFilledAt = time.ToUniversalTime();
            Emit(cmd, eventType, role, quantity, price, null, orderId, order.OrderState.ToString(), executionId);
        }

        private string Role(Order order, string signalName)
        {
            if (order == null) return "UNKNOWN";
            if (order.Name == signalName) return "ENTRY";
            if (order.Name.IndexOf("Profit", StringComparison.OrdinalIgnoreCase) >= 0) return "TARGET";
            if (order.Name.IndexOf("Stop", StringComparison.OrdinalIgnoreCase) >= 0) return "STOP";
            return "FLATTEN";
        }

        private void EmitHeartbeat()
        {
            if ((DateTime.UtcNow - lastHeartbeat).TotalSeconds < 5)
                return;
            lastHeartbeat = DateTime.UtcNow;
            JObject heartbeat = new JObject();
            heartbeat["command_id"] = "__bridge__";
            heartbeat["objective_id"] = "__bridge__";
            heartbeat["session_date"] = DateTime.UtcNow.ToString("yyyy-MM-dd");
            Emit(heartbeat, "BRIDGE_HEARTBEAT", null, 0, 0, null);
        }

        private void Emit(JObject cmd, string eventType, string role, int fillQty,
            double fillPrice, string reason, string orderId = null, string orderState = null,
            string executionId = null)
        {
            lock (ioLock)
            {
                string eventAt = DateTime.UtcNow.ToString("o");
                string eventId = "evt_" + Guid.NewGuid().ToString("N");
                JObject evt = new JObject();
                evt["schema_version"] = "v1";
                evt["event_id"] = eventId;
                evt["event_type"] = eventType;
                evt["event_at"] = eventAt;
                evt["command_id"] = (string)cmd["command_id"];
                evt["signal_id"] = (string)cmd["signal_id"];
                evt["objective_id"] = (string)cmd["objective_id"];
                evt["session_date"] = (string)cmd["session_date"];
                evt["account"] = Account == null ? null : Account.Name;
                evt["instrument"] = Instrument == null ? null : Instrument.FullName;
                evt["action"] = (string)cmd["action"];
                evt["role"] = role;
                evt["fill_qty"] = fillQty;
                evt["fill_price"] = fillQty > 0 ? (JToken)fillPrice : JValue.CreateNull();
                evt["reason"] = reason;
                evt["order_id"] = orderId;
                evt["order_state"] = orderState;
                evt["execution_id"] = executionId;
                evt["market_position"] = Position == null ? null : Position.MarketPosition.ToString();
                evt["position_qty"] = Position == null ? 0 : Position.Quantity;
                string inbox = Path.Combine(BridgeDirectory, "inbox");
                string target = Path.Combine(inbox, eventId + ".json");
                string temp = Path.Combine(inbox, "." + eventId + ".tmp");
                File.WriteAllText(temp, evt.ToString());
                File.Move(temp, target);
            }
        }
    }
}