"""Exercise a running recorded console and its MLflow UI in a real browser.

Install Playwright and its Chromium browser separately from the small viewer
runtime. Start ``python -m src.showcase serve`` first, then run this script.
No market/database service or external website is contacted by the console.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import urlopen

from playwright.sync_api import Browser, Page, expect, sync_playwright


def _wait_ready(url: str, timeout: float = 60.0) -> None:
    """Allow a concurrently started CLI to bring both local services online."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError as exc:
            last_error = exc
        time.sleep(0.4)
    raise RuntimeError(f"Service did not become ready within {timeout:g}s: {url}; {last_error}")


def _origin(url: str) -> tuple[str, str]:
    parsed = urlsplit(url)
    return parsed.scheme, parsed.netloc


def _screenshot(page: Page, output: Path | None, name: str) -> None:
    if output is not None:
        page.screenshot(path=str(output / f"{name}.png"), full_page=True)


def _console(browser: Browser, url: str, output: Path | None) -> dict[str, Any]:
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    external_requests: list[str] = []
    page_errors: list[str] = []
    expected_origin = _origin(url)

    def inspect_request(request: Any) -> None:
        if request.url.startswith(("http://", "https://")) and _origin(request.url) != expected_origin:
            external_requests.append(request.url)

    context.on("request", inspect_request)
    page = context.new_page()
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    try:
        page.goto(url, wait_until="domcontentloaded")
        expect(page.get_by_role("heading", name="Robin: a Transformer, end to end")).to_be_visible()
        response = context.request.get(url.rstrip("/") + "/api/showcase")
        assert response.ok, f"Showcase API failed: {response.status}"
        payload = response.json()
        assert payload["summary"]["evidence_kind"] == "recorded_example"
        assert payload["summary"]["n_rows"] == 3000
        assert payload["viewer"]["read_only"] is True
        expected_link = payload["viewer"]["mlflow_url"]
        expect(page.locator("#mlflow-link")).to_have_attribute("href", expected_link)
        tabs = ["dashboard", "data", "models", "backtest", "evaluation", "artifacts", "live"]
        expect(page.locator("[data-nav]")).to_have_count(len(tabs))
        _screenshot(page, output, "01-dashboard")

        def navigate(tab: str, heading: str) -> None:
            page.locator(f'[data-nav="{tab}"]').click()
            expect(page.get_by_role("heading", name=heading, exact=True)).to_be_visible()

        navigate("data", "Data / Features")
        expect(page.locator("svg.chart")).to_have_count(2)
        page.select_option("#split-choice", "holdout")
        page.select_option("#feature-choice", "wf_momentum_12")
        expect(page.get_by_role("heading", name="wf_momentum_12", exact=True)).to_be_visible()
        expect(page.locator(".chart-caption").first).to_contain_text("300 actual observations")
        page.check("#outcomes-choice")
        expect(page.locator("svg.chart")).to_have_count(3)
        expect(page.get_by_role("heading", name="Future outcomes · never model inputs")).to_be_visible()
        _screenshot(page, output, "02-data-features-outcomes")
        page.uncheck("#outcomes-choice")
        expect(page.locator("svg.chart")).to_have_count(2)

        navigate("models", "Model design comparison")
        selected = next(model for model in payload["models"] if model["selected"])
        other = next(model for model in payload["models"] if not model["selected"])
        page.select_option("#model-choice", selected["design_id"])
        page.select_option("#prediction-split", "holdout")
        expect(page.locator(".chart-caption")).to_contain_text("289 actual observations")
        _screenshot(page, output, "03-models-holdout")
        page.select_option("#model-choice", other["design_id"])
        expect(page.locator("#prediction-split option")).to_have_count(1)
        expect(page.locator(".chart-caption")).to_contain_text("435 actual observations")

        navigate("backtest", "Synthetic backtest replay")
        page.select_option("#model-choice", selected["design_id"])
        expect(page.locator("tbody tr")).to_have_count(70)
        expect(page.locator("svg.chart")).to_have_count(2)
        _screenshot(page, output, "04-backtest")
        page.select_option("#model-choice", other["design_id"])
        expect(page.locator("tbody tr")).to_have_count(75)

        navigate("evaluation", "Evaluation / Promotion")
        for heading in ("Validation gates", "Locked holdout gates"):
            gate_card = page.locator(".card").filter(has=page.get_by_role("heading", name=heading, exact=True))
            expect(gate_card.get_by_role("cell", name="PASS", exact=True)).to_have_count(8)
            expect(gate_card.get_by_role("cell", name="FAIL", exact=True)).to_have_count(0)
        expect(page.get_by_role("cell", name="Rejected as expected", exact=True)).to_have_count(2)
        expect(page.get_by_text("This viewer: recorded example", exact=True)).to_be_visible()
        _screenshot(page, output, "05-evaluation-promotion")

        navigate("artifacts", "Artifacts & tracking")
        expect(page.get_by_role("link", name="synthetic.parquet", exact=True)).to_be_visible()
        expect(page.locator("tbody tr")).to_have_count(len(payload["artifacts"]))
        _screenshot(page, output, "06-artifacts")
        navigate("live", "Live platform capabilities")
        expect(page.get_by_role("button", name="Connect market feed")).to_be_disabled()
        expect(page.get_by_role("button", name="Launch operational job")).to_be_disabled()
        assert not page_errors, f"Console JavaScript errors: {page_errors}"
        assert not external_requests, f"Unexpected external console requests: {external_requests}"
        return {"tabs": tabs, "dataset_rows": 3000, "validation_rows": 435,
                "holdout_rows": 289, "ledger_rows": [70, 75],
                "gate_passes": {"validation": 8, "holdout": 8},
                "page_errors": page_errors, "external_requests": external_requests,
                "source_run_id": payload["summary"]["source_run_id"],
                "mlflow": payload.get("mlflow", {})}
    except Exception:
        _screenshot(page, output, "console-failure")
        raise
    finally:
        context.close()


def _mlflow(browser: Browser, url: str, console: dict[str, Any], output: Path | None) -> dict[str, Any]:
    """Wait for rendered data, never networkidle (MLflow performs polling)."""
    context = browser.new_context(viewport={"width": 1600, "height": 1000})
    page = context.new_page()
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    try:
        page.goto(url, wait_until="domcontentloaded")
        experiment = page.get_by_role("link", name="Robin recorded Transformer example", exact=True)
        expect(experiment).to_be_visible(timeout=30000)
        experiment_id = console["mlflow"].get("experiment_id")
        assert experiment_id, "Use a prepared example with imported MLflow lineage"
        # The direct runs route avoids MLflow 3's default GenAI overview.
        page.goto(url.rstrip("/") + f"/#/experiments/{experiment_id}/runs", wait_until="domcontentloaded")
        expect(page.get_by_text("3 matching runs", exact=True)).to_be_visible(timeout=30000)
        parent_name = "Recorded: " + console["source_run_id"]
        parent = page.get_by_role("link", name=parent_name, exact=True)
        expect(parent).to_be_visible()
        # MLflow folds nested design runs under their parent by default.
        parent.locator('xpath=ancestor::*[@role="gridcell"]').get_by_role("button").click()
        names = [parent_name, "Recorded: transformer-depth-1", "Recorded: transformer-depth-2"]
        for name in names:
            expect(page.get_by_role("link", name=name, exact=True)).to_be_visible()
        _screenshot(page, output, "07-mlflow-lineage")
        return {"experiment_id": str(experiment_id), "visible_run_names": names,
                "matching_runs": 3, "page_errors_diagnostic": page_errors}
    except Exception:
        _screenshot(page, output, "mlflow-failure")
        raise
    finally:
        context.close()


def main() -> int:
    """Run browser checks; optionally save screenshots and machine-readable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8700", help="Running showcase console")
    parser.add_argument("--mlflow-url", default="http://127.0.0.1:5000", help="Its local MLflow server")
    parser.add_argument("--output", type=Path, help="New directory for screenshots and browser-results.json")
    parser.add_argument("--chromium", type=Path, help="Optional Chromium executable; default: Playwright-managed Chromium")
    args = parser.parse_args()
    for url in (args.url, args.mlflow_url):
        if urlsplit(url).scheme not in {"http", "https"} or not urlsplit(url).netloc:
            parser.error("Service URLs must be HTTP(S) URLs")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result: dict[str, Any] = {"passed": False, "console_url": args.url, "mlflow_url": args.mlflow_url}
    try:
        print("████░░░░░░░░░░░░░░░░ Waiting for console and MLflow", file=sys.stderr, flush=True)
        _wait_ready(args.url.rstrip("/") + "/api/showcase/health")
        _wait_ready(args.mlflow_url.rstrip("/") + "/health")
        with sync_playwright() as playwright:
            launch: dict[str, Any] = {"headless": True}
            if args.chromium:
                launch["executable_path"] = str(args.chromium)
            browser = playwright.chromium.launch(**launch)
            try:
                result["console"] = _console(browser, args.url, args.output)
                result["mlflow"] = _mlflow(browser, args.mlflow_url, result["console"], args.output)
            finally:
                browser.close()
        result["passed"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        (args.output / "browser-results.json").write_text(serialized)
    print(serialized)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
