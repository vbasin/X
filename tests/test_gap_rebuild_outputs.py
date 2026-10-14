"""Historical gap outputs preserve bytes and join only knowable symbol terms."""
from pathlib import Path

import pandas as pd
import pytest

from scripts.data.rebuild_gap_outputs import (
    causal_option_inputs, closed_context_sources, closed_rows, compute_causal_greeks,
    definition_receipt_paths, exclude_source_gap_bars, load_vendor_snapshots, numeric_tape_side, preserve_then_write,
    option_upsert_statement, publish_native_candidate, replace_window, source_quality, utc,
    evaluate_option_release, option_product_window,
    rebuild_day,
    option_candidate_scope,prepare_option_publication_attempt,file_sha256,
    compressed_append_only,
    publish_candidates,validate_option_parquets,
    classify_sql_payload_differences,
    definition_record_availability,
)

pytestmark = pytest.mark.acquisition


def test_features_only_records_blocker_without_engine_database_or_vendor_reads(tmp_path,monkeypatch):
    import builtins
    import scripts.data.rebuild_gap_outputs as module
    original_import = builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name in {"psycopg","src.features.engine","src.common.db"}:
            raise AssertionError(f"blocked features must not import {name}")
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded_import)
    monkeypatch.setattr(module,"load_vendor_snapshots",lambda _: pytest.fail("blocked features read vendor data"))
    result = rebuild_day(("2026-10-06","2026-10-06T17:57:00Z",str(tmp_path),["absent.dbn"],["features"],[]))
    product = result["products"]["features_5s"]
    assert result["status"] == "blocked"
    assert product["publishable"] is False and product["rows"] == 0
    assert any("30s" in reason for reason in product["reasons"])
    assert (tmp_path / "candidates" / "2026-10-06" / "manifest.json").exists()
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("bad",["fraction","null_id","null_ts","duplicate"])
def test_option_copy_rejects_lossy_or_null_keys(bad):
    frame = pd.DataFrame({"ts":[_ts()],"instrument_id":[123]})
    if bad == "fraction":
        frame["ts"] = frame["ts"]+pd.Timedelta(milliseconds=1)
    elif bad == "null_id":
        frame["instrument_id"] = None
    elif bad == "null_ts":
        frame["ts"] = pd.NaT
    else:
        frame = pd.concat([frame,frame])
    with pytest.raises(ValueError,match="non-null, unique whole seconds"):
        option_candidate_scope(frame,["instrument_id","ts"],_ts(-1),_ts(2))


def test_fresh_publication_attempt_keeps_snapshots_and_checks_candidate_identity(tmp_path):
    import json
    source = tmp_path / "original"
    source.mkdir()
    paths = {}
    for name in ["greeks","surface"]:
        path = source / f"{name}.parquet"
        pd.DataFrame({"ts":[_ts()],"instrument_id":[123]}).to_parquet(path,index=False)
        paths[name] = {"path":str(path),"rows":1,"publishable":True}
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({"days":[{"date":"2026-10-06","products":paths}]}))
    saved = source / "preserved_db" / "2026-10-06" / "greeks.parquet"
    saved.parent.mkdir(parents=True)
    pd.DataFrame({"old":[1]}).to_parquet(saved)
    before = file_sha256(saved)
    retry = tmp_path / "retry"
    receipt = prepare_option_publication_attempt(source,retry)
    assert receipt["prior_snapshots"][str(saved.resolve())] == before
    assert file_sha256(saved) == before
    assert (retry / "manifest.json").read_bytes() == manifest.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_option_publication_attempt(source,retry)
    pd.DataFrame({"changed":[1]}).to_parquet(paths["greeks"]["path"])
    with pytest.raises(ValueError,match="candidate changed"):
        prepare_option_publication_attempt(retry,tmp_path / "second_retry")
    assert not (tmp_path / "second_retry").exists()


def test_compressed_overlap_requires_append_only_without_existing_row_population():
    assert compressed_append_only([("internal","chunk",_ts(),_ts(5),True)])
    assert compressed_append_only([("internal","a",_ts(),_ts(5),False),("internal","b",_ts(5),_ts(10),True)])
    assert not compressed_append_only([("internal","chunk",_ts(),_ts(5),False)])
    assert not compressed_append_only([])


def test_greek_file_replacement_retains_other_ids_inside_window():
    old = pd.DataFrame({"ts":[_ts(),_ts(),_ts(10)],"instrument_id":[123,456,123],"value":[1,2,3]})
    new = pd.DataFrame({"ts":[_ts()],"instrument_id":[123],"value":[4]})
    result = replace_window(old,new,_ts(-1),_ts(2),timestamp="ts",instrument_ids=[123])
    assert sorted(result["value"]) == [2,3,4]
    assert result.loc[result["instrument_id"].eq(456),"value"].tolist() == [2]


def test_independent_parquet_validation_checks_payloads_keys_and_outside_scopes(tmp_path):
    import json
    from src.common.data_paths import options_greeks_dir,options_surface_dir
    run = tmp_path / "attempt"
    run.mkdir()
    products = {}
    for name in ["greeks","surface"]:
        frame = pd.DataFrame({"ts":[_ts()],"value":[4]})
        if name == "greeks":
            frame["instrument_id"] = 123
        else:
            frame["n_legs_filled"] = pd.Series([4],dtype="int16")
        path = run / f"{name}.parquet"
        frame.to_parquet(path,index=False)
        products[name] = {"path":str(path),"rows":1,"publishable":True,"scope_start":_ts(-1).isoformat(),"scope_end":_ts(2).isoformat()}
    greek_path = options_greeks_dir() / "greeks_2026-10-06.parquet"
    greek_path.parent.mkdir(parents=True)
    pd.DataFrame({"ts":[_ts(),_ts(10)],"value":[2,3],"instrument_id":[456,123]}).to_parquet(greek_path,index=False)
    surface_path = options_surface_dir() / "features_2026-10-06.parquet"
    surface_path.parent.mkdir(parents=True)
    pd.DataFrame({"ts":[_ts(10)],"value":[3],"n_legs_filled":pd.Series([4],dtype="int64")}).set_index("ts").to_parquet(surface_path)
    (run / "manifest.json").write_text(json.dumps({"days":[{"date":"2026-10-06","products":products}]}))
    publish_candidates(run,["greeks","surface"])
    live = pd.read_parquet(surface_path)
    live.loc[_ts(11)] = live.loc[_ts(10)]
    live.to_parquet(surface_path)
    receipt = validate_option_parquets(run)
    rows = {row["product"]:row for row in receipt["results"]}
    assert rows["greeks"]["candidate_keys"] == rows["surface"]["candidate_keys"] == 1
    assert rows["greeks"]["outside_scope_rows_preserved"] == 2
    assert rows["surface"]["outside_scope_rows_preserved"] == 1
    assert rows["surface"]["new_live_outside_scope_rows"] == 1
    assert all(row["payload_differences"] == 0 and row["duplicate_keys"] == 0 for row in rows.values())
    changed = pd.read_parquet(surface_path)
    changed.loc[_ts(),"n_legs_filled"] = 5
    changed.to_parquet(surface_path)
    retry = tmp_path / "changed_attempt"
    retry.mkdir()
    for name in ["manifest.json","publication.json"]:
        (retry / name).write_bytes((run / name).read_bytes())
    with pytest.raises(RuntimeError,match="canonical payload differs"):
        validate_option_parquets(retry)


def test_sql_missing_numeric_representation_is_separate_from_material_differences():
    rows = [{"changes":{"iv":{"old":"NaN","new":None}}},
            {"changes":{"iv":{"old":0.3,"new":0.4}}},
            {"changes":{"symbol":{"old":"NaN","new":None}}}]
    result = classify_sql_payload_differences(rows,{"iv"})
    assert result == {"exact_sql_differing_rows":3,"material_differing_rows":2,"legacy_missing_representation_only_rows":1}


def test_ordinary_definition_waits_for_future_vendor_receive_timestamp():
    from types import SimpleNamespace
    record = SimpleNamespace(ts_event=_ts(-1).value,ts_recv=_ts(1).value)
    available,snapshot = definition_record_availability(record)
    assert available == _ts(1) and not snapshot
    defs = _defs(_ts(-1)).assign(available_at=available)
    joined,counts = causal_option_inputs(_bbo(),defs,_und())
    assert joined.empty and counts["missing_definition"] == 1


def test_weekday_midnight_definition_snapshot_retains_known_publication_timestamp():
    from types import SimpleNamespace
    event = pd.Timestamp("2026-10-04T13:00:00Z")
    received = pd.Timestamp("2026-10-06T00:00:00Z")
    available,snapshot = definition_record_availability(SimpleNamespace(ts_event=event.value,ts_recv=received.value))
    assert available == event and snapshot


def _ts(second=0):
    return pd.Timestamp("2026-10-06T13:00:00Z") + pd.Timedelta(seconds=second)


def _defs(timestamp=None, **kw):
    row = dict(ts_event=_ts(-3600) if timestamp is None else timestamp,
               instrument_id=999, symbol="Q1BV6 C25000",instrument_class="C",
               strike_price=25000.0,expiration=pd.Timestamp("2026-10-06T20:00:00Z"),
               underlying="NQZ6",multiplier=20.0,source="databento")
    row.update(kw)
    return pd.DataFrame([row])


def _bbo():
    return pd.DataFrame([dict(ts_recv=_ts(),instrument_id=123,symbol="Q1BV6 C25000",bid_px=99.,ask_px=101.)])


def _und():
    return pd.DataFrame([dict(ts_recv=_ts(-1),symbol="NQZ6",bid_px=24999.,ask_px=25001.),
                         dict(ts_recv=_ts(.5),symbol="NQZ6",bid_px=25999.,ask_px=26001.)])


def test_terms_join_raw_symbol_and_keep_quote_id():
    joined,counts = causal_option_inputs(_bbo(),_defs(),_und())
    assert joined["instrument_id"].tolist() == [123]
    assert joined["nq_price"].tolist() == [25000.]
    assert not counts["missing_definition"]


def test_asof_joins_normalize_different_timestamp_resolutions():
    bbo = _bbo()
    bbo["ts_recv"] = bbo["ts_recv"].dt.as_unit("us")
    und = _und()
    und["ts_recv"] = und["ts_recv"].dt.as_unit("ms")
    joined,counts = causal_option_inputs(bbo,_defs(),und)
    assert joined["nq_price"].tolist() == [25000.]
    assert counts["missing_definition"] == counts["missing_underlying"] == 0


def test_future_definition_cannot_fill_earlier_quote():
    joined,counts = causal_option_inputs(_bbo(),_defs(_ts(1)),_und())
    assert joined.empty
    assert counts["missing_definition"] == 1


def test_changed_vendor_terms_are_selected_asof_quote():
    defs = pd.concat([_defs(),_defs(_ts(1),strike_price=26000.)])
    quotes = pd.concat([_bbo(),_bbo().assign(ts_recv=_ts(2))])
    joined,_ = causal_option_inputs(quotes,defs,_und())
    assert joined["strike_price"].tolist() == [25000.,26000.]


@pytest.mark.parametrize("fields",[{"source":"synthetic"},{"underlying":""},{"strike_price":float("nan")},{"available_at":pd.NaT}])
def test_invalid_or_synthetic_terms_are_missing(fields):
    joined,counts = causal_option_inputs(_bbo(),_defs(**fields),_und())
    assert joined.empty
    assert counts["missing_definition"] == 1


def test_future_within_second_underlying_never_floored_back():
    joined,counts = causal_option_inputs(_bbo(),_defs(),_und().iloc[1:])
    assert joined.empty
    assert counts["missing_underlying"] == 1


def test_exact_timestamp_definition_and_underlying_are_inclusive():
    joined,_ = causal_option_inputs(_bbo(),_defs(_ts()),_und().iloc[:1].assign(ts_recv=_ts()))
    assert len(joined) == 1


def test_known_source_gap_blocks_prices_until_fresh_post_gap_underlying():
    quotes = pd.concat([_bbo().assign(ts_recv=_ts(second)) for second in (0,2,5,6)])
    underlying = pd.concat([_und().iloc[:1],_und().iloc[:1].assign(ts_recv=_ts(6))])
    joined,counts = causal_option_inputs(quotes,_defs(),underlying,
                                        source_gaps=[(_ts(1).isoformat(),_ts(5).isoformat())])
    assert joined["ts_recv"].tolist() == [_ts(),_ts(6)]
    assert counts["source_gap_price_exclusions"] == 2


def test_native_bar_interval_gap_boundaries_and_source_quality():
    frame = pd.DataFrame({"bar_end_ts":[_ts(),_ts(15),_ts(30),_ts(45)],"value":[1,2,3,4]})
    gaps = [(_ts().isoformat(),_ts(30).isoformat())]
    actual,removed = exclude_source_gap_bars(frame,"15s",gaps)
    assert actual["value"].tolist() == [1,4]
    assert removed == 2
    quality = source_quality(_ts(-1),_ts(45),gaps)
    assert not quality["source_complete"]
    assert len(quality["unrecoverable_source_gaps"]) == 1
    assert source_quality(_ts(30),_ts(45),gaps)["source_complete"]


def test_context_source_filter_prevents_partial_hours_entering_rolls():
    frame = pd.DataFrame({"bucket":pd.to_datetime(["2026-10-05T21:00:00Z","2026-10-05T22:00:00Z","2026-10-06T00:00:00Z","2026-10-06T01:00:00Z"])})
    gaps = [("2026-10-05T22:50:00Z","2026-10-06T00:00:00Z")]
    actual,removed = closed_context_sources(frame,"1h","2026-10-06T01:30:00Z",gaps)
    assert actual.bucket.tolist() == [pd.Timestamp("2026-10-05T21:00:00Z"),pd.Timestamp("2026-10-06T00:00:00Z")]
    assert removed == 2


def test_context_daily_close_observes_dst_and_withholds_unclosed_day():
    frame = pd.DataFrame({"bucket":pd.to_datetime(["2026-10-31T22:00:00Z"])})
    before,_ = closed_context_sources(frame,"1d","2026-11-01T22:30:00Z",[])
    after,_ = closed_context_sources(frame,"1d","2026-11-01T23:00:00Z",[])
    assert before.empty
    assert len(after) == 1


def test_native_source_grids_do_not_fabricate_empty_outage_bars():
    from src.features.bar_features import compute_bar_features
    from src.features.native_tf_1m import candles_1m_from_tbbo
    from src.features.native_tf_5m import candles_5m_from_tbbo
    ticks = pd.DataFrame({"ts_event":pd.to_datetime(["2026-10-05T22:49:00Z","2026-10-06T00:00:00Z"]),
        "price":[25000.,25001.],"size":[1,1],"side":["B","A"],
        "bid_px":[24999.,25000.],"ask_px":[25001.,25002.],"bid_sz":[1,1],"ask_sz":[1,1]})
    assert len(compute_bar_features(ticks,"15s")) == 2
    assert len(candles_1m_from_tbbo(ticks)) == 2
    assert len(candles_5m_from_tbbo(ticks)) == 2


def test_saved_definition_receipts_resolve_only_vendor_snapshot_paths(tmp_path):
    import json
    file = tmp_path / "records.dbn"
    file.write_bytes(b"DBN")
    (tmp_path / "one.json").write_text(json.dumps({"params":{"schema":"definition"},"receipt":{"path":str(file)}}))
    (tmp_path / "two.json").write_text(json.dumps({"params":{"schema":"bbo-1s"},"receipt":{"path":"not-read"}}))
    assert definition_receipt_paths(tmp_path) == [file]


def test_raw_definition_availability_keeps_vendor_nanoseconds(monkeypatch,tmp_path):
    import databento as db
    from types import SimpleNamespace
    record = SimpleNamespace(raw_symbol="Q1BV6 C25000",ts_event=_ts().value+1,
        instrument_id=999,instrument_class="C",strike_price=25000*10**9,
        expiration=pd.Timestamp("2026-10-06T20:00:00Z").value,underlying="NQZ6",unit_of_measure_qty=20*10**9)
    monkeypatch.setattr(db.DBNStore,"from_file",lambda _: [record])
    definitions = load_vendor_snapshots([tmp_path / "mock.dbn"])
    assert definitions.ts_event.iloc[0].value == _ts().value+1
    joined,counts = causal_option_inputs(_bbo(),definitions,_und())
    assert joined.empty and counts["missing_definition"] == 1


def test_tape_sides_accept_categorical_with_no_neutral_category():
    values = pd.Series(pd.Categorical(["B","A",None],categories=["B","A"]))
    assert numeric_tape_side(values).tolist() == [1,-1,0]


def test_compressed_output_sql_appends_only_absent_keys_with_bounded_lookup():
    statement = option_upsert_statement("nq_option_greeks_1s",["ts","instrument_id","symbol"],
                                        ["instrument_id","ts"],append_only=True).as_string()
    assert "NOT EXISTS" in statement
    assert 't."instrument_id"=s."instrument_id"' in statement
    assert "ON CONFLICT" in statement and "DO NOTHING" in statement
    assert statement.count("%s") == 4
    assert "UPDATE" not in statement
    assert "DELETE" not in statement and "TRUNCATE" not in statement


def test_uncompressed_output_upsert_updates_only_changed_bounded_rows():
    statement = option_upsert_statement("nq_option_surface_1s",["ts","nq_price"],["ts"],append_only=False).as_string()
    assert "DO UPDATE" in statement and "IS DISTINCT FROM" in statement
    assert "kept.ts >= %s AND kept.ts < %s" in statement
    assert statement.count("%s") == 4
    assert "DELETE" not in statement and "TRUNCATE" not in statement


def test_off_grid_greeks_labeled_when_quote_is_knowable():
    joined,_ = causal_option_inputs(_bbo().assign(ts_recv=_ts(.5)),_defs(),_und().iloc[:1])
    frame = compute_causal_greeks(joined)
    assert len(frame) == 1
    assert frame["ts"].iloc[0] == _ts(1)
    assert frame["instrument_id"].iloc[0] == 123


@pytest.mark.parametrize("expiry_second",[.75,1.0])
def test_forward_availability_label_at_or_after_expiration_is_excluded(expiry_second):
    joined,_ = causal_option_inputs(_bbo().assign(ts_recv=_ts(.5)),
                                    _defs(expiration=_ts(expiry_second)),_und().iloc[:1])
    assert compute_causal_greeks(joined).empty


def test_0dte_product_scope_excludes_raw_gap_after_vendor_expiry():
    gaps=[("2026-10-05T22:50:00Z","2026-10-06T00:00:00Z")]
    scope=option_product_window("2026-10-05T05:00:00Z","2026-10-06T05:00:00Z",
        "2026-10-05T20:00:00Z","2026-10-06T17:57:00Z",gaps)
    assert scope["scope_end"] == "2026-10-05T20:00:00+00:00"
    assert scope["source_complete"]
    assert not source_quality("2026-10-05T05:00:00Z","2026-10-06T05:00:00Z",gaps)["source_complete"]


def test_release_manifest_preserves_original_raw_day_evidence(tmp_path):
    import json
    path=tmp_path / "candidate.parquet"
    pd.DataFrame({"ts":[pd.Timestamp("2026-10-05T13:00:00Z")]}).to_parquet(path,index=False)
    original={"asof":"2026-10-06T17:57:00Z","days":[{"date":"2026-10-05",
        "option_coverage":{"quotes":1,"missing_definition":0,"missing_underlying":0,
                           "source_complete":False,"source_gap_price_exclusions":0},
        "products":{name:{"path":str(path),"rows":1,"publishable":False} for name in ("greeks","surface")}}]}
    raw=json.dumps(original)
    (tmp_path / "manifest.json").write_text(raw)
    definitions=_defs(expiration=pd.Timestamp("2026-10-05T20:00:00Z"))
    actual=evaluate_option_release(tmp_path,definitions,[("2026-10-05T22:50:00Z","2026-10-06T00:00:00Z")])
    assert (tmp_path / "manifest.json").read_text() == raw
    day=actual["days"][0]
    assert day["products"]["greeks"]["publishable"]
    assert not day["products"]["greeks"]["raw_calendar_day_source_complete"]
    assert day["products"]["greeks"]["scope_end"] == "2026-10-05T20:00:00+00:00"


def test_id_collision_across_raw_symbols_is_not_silently_collapsed():
    joined,_ = causal_option_inputs(_bbo(),_defs(),_und())
    joined = pd.concat([joined,joined.assign(symbol="Q1BV6 P25000")])
    with pytest.raises(ValueError,match="multiple raw symbols"):
        compute_causal_greeks(joined)


def test_crossed_quote_is_not_priced():
    joined,_ = causal_option_inputs(_bbo().assign(bid_px=101.,ask_px=99.),_defs(),_und())
    assert compute_causal_greeks(joined).empty


def test_closed_rows_drop_unfinished_bar():
    frame = pd.DataFrame({"bar_end_ts":[_ts(),_ts(15)],"value":[1,2]})
    assert closed_rows(frame,_ts(5))["value"].tolist() == [1]


def test_replace_window_retains_adjacent_sessions_and_checks_scope():
    old = pd.DataFrame({"ts":[_ts(-1),_ts(),_ts(1),_ts(2)],"value":[1,2,3,4]})
    new = pd.DataFrame({"ts":[_ts()],"value":[8]})
    merged = replace_window(old,new,_ts(),_ts(2),timestamp="ts")
    assert merged["value"].tolist() == [1,8,4]
    with pytest.raises(ValueError):
        replace_window(old,new,_ts(1),_ts(2),timestamp="ts")


def test_preserve_before_replacing_and_refuse_snapshot_reuse(tmp_path):
    target = tmp_path / "live" / "part.parquet"
    target.parent.mkdir()
    pd.DataFrame({"value":[1]}).to_parquet(target,index=False)
    original = target.read_bytes()
    preserved = tmp_path / "preserved"
    preserve_then_write(pd.DataFrame({"value":[2]}),target,preserved)
    assert (preserved / target.name).read_bytes() == original
    assert pd.read_parquet(target)["value"].tolist() == [2]
    with pytest.raises(FileExistsError):
        preserve_then_write(pd.DataFrame({"value":[3]}),target,preserved)
    assert pd.read_parquet(target)["value"].tolist() == [2]


def test_native_publish_keeps_month_neighbors_without_unlink(tmp_path):
    directory = tmp_path / "month"
    directory.mkdir()
    original = directory / "features_15s_part1.parquet"
    old = pd.DataFrame({"bar_end_ts":[_ts(-15),_ts(),_ts(15)],"value":[1,2,3]})
    old.to_parquet(original,index=False)
    original_bytes = original.read_bytes()
    candidate = pd.DataFrame({"bar_end_ts":[_ts()],"value":[8]})
    preserved = tmp_path / "preserved"
    changes = publish_native_candidate(candidate,directory,preserved,timeframe="15s",start=_ts(),end=_ts(1))
    assert original.exists()
    assert (preserved / original.name).read_bytes() == original_bytes
    assert pd.read_parquet(original)["value"].tolist() == [1,3]
    assert pd.read_parquet(directory / "features_15s_part2.parquet")["value"].tolist() == [8]
    assert len(changes) == 2


def test_naive_cutoff_is_rejected():
    with pytest.raises(ValueError):
        utc("2026-10-06T13:00:00")
