"""Tests for the public command-line surface."""

from typer.testing import CliRunner

from chronoslob import __version__
from chronoslob.cli import app

EXPECTED_COMMANDS = {
    "analyse-fi2010-execution-v3",
    "analyse-fi2010-feature-ablations",
    "analyse-fi2010-ssl-results",
    "analyse-fi2010-ssl-v2-results",
    "analyse-fi2010-uncertainty",
    "audit-fi2010-features",
    "build-execution-centrepiece",
    "build-fi2010-ablation-figures",
    "build-fi2010-execution-v3",
    "build-fi2010-figures",
    "build-paper-plots",
    "build-paper-report",
    "convert-fi2010-official",
    "doctor",
    "event-log-to-features",
    "init-run",
    "inspect-analysis",
    "inspect-baselines",
    "inspect-binance-replay",
    "inspect-calibration",
    "inspect-deeplob",
    "inspect-event-log",
    "inspect-event-tokens",
    "inspect-execution-validation",
    "inspect-experiment-artifacts",
    "inspect-features-fi2010",
    "inspect-fi2010",
    "inspect-fi2010-multifold",
    "inspect-fi2010-neural-plan",
    "inspect-labels-fi2010",
    "inspect-multitask",
    "inspect-paper-experiment",
    "inspect-paper-report",
    "inspect-split",
    "inspect-ssl",
    "inspect-system-benchmarks",
    "inspect-torch-dataset",
    "inspect-transformer",
    "prepare-fi2010-benchmark",
    "prepare-fi2010-multifold",
    "replay-binance-l2-sample",
    "run-baseline-smoke",
    "run-calibration-smoke",
    "run-deeplob-smoke",
    "run-execution-validation-smoke",
    "run-fi2010-brutal-ablations",
    "run-fi2010-execution-v2",
    "run-fi2010-feature-ablations",
    "run-fi2010-multifold-classical",
    "run-fi2010-neural-benchmark",
    "run-fi2010-neural-full-grid",
    "run-fi2010-neural-proper-training-subset",
    "run-fi2010-ssl-neural-benchmark",
    "run-fi2010-ssl-v2-benchmark",
    "run-multitask-smoke",
    "run-paper-ablations",
    "run-paper-experiment",
    "run-robustness-analysis-smoke",
    "run-ssl-smoke",
    "run-synthetic-lob-benchmark",
    "run-system-benchmarks",
    "run-transformer-smoke",
    "verify-fi2010-local",
    "version",
}


def _registered_command_names() -> list[str]:
    return [
        command.name or command.callback.__name__.replace("_", "-")
        for command in app.registered_commands
    ]


def test_public_commands_are_registered_once() -> None:
    command_names = _registered_command_names()

    assert set(command_names) == EXPECTED_COMMANDS
    assert len(command_names) == len(set(command_names))


def test_help_lists_representative_command_domains() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command_name in (
        "inspect-event-log",
        "run-fi2010-neural-benchmark",
        "analyse-fi2010-uncertainty",
        "inspect-binance-replay",
        "run-calibration-smoke",
    ):
        assert command_name in result.stdout


def test_version_command_reports_package_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
