"""Tests for deciding which months of a channel need exporting."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts.channel_classifier import ChannelType
from scripts.discord_api import ChannelActivity, DiscordApiError
from scripts.export_channels import (
    ChannelExportContext,
    _process_single_channel,
    export_all_channels,
)
from scripts.export_plan import (
    PHASE_BACKFILL,
    PHASE_CURRENT,
    CacheEntry,
    ExportCache,
    plan_months,
    published_has_message,
)

DISCORD_EPOCH_MS = 1420070400000
NOW = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
CURRENT = "2026-11"


def flake(year: int, month: int, day: int = 15, hour: int = 12) -> str:
    """A Discord snowflake ID created at the given UTC time."""
    ms = int(datetime(year, month, day, hour, tzinfo=UTC).timestamp() * 1000)
    return str((ms - DISCORD_EPOCH_MS) << 22)


CHANNEL_ID = flake(2026, 8, 1)


def publish(
    channel_dir: Path, month: str, message_ids: list[str], exported_at: datetime | None = None
) -> None:
    """Write a published month export holding the given message IDs."""
    month_dir = channel_dir / month
    month_dir.mkdir(parents=True, exist_ok=True)
    stamp = (exported_at or NOW).isoformat()
    export = {"exportedAt": stamp, "messages": [{"id": m} for m in message_ids]}
    (month_dir / f"{month}.json").write_text(json.dumps(export), encoding="utf-8")


def plan(  # noqa: PLR0913  # Each argument is one independent fact about the channel
    phase: str,
    newest: str | None,
    channel_dir: Path,
    *,
    entry: CacheEntry | None = None,
    completed: set[str] | None = None,
    refresh_hours: float = 24.0,
) -> list[str]:
    return plan_months(
        phase,
        CHANNEL_ID,
        ChannelActivity(0, newest),
        entry,
        public_channel_dir=channel_dir,
        completed=completed or set(),
        current_month=CURRENT,
        now=NOW,
        refresh_hours=refresh_hours,
    )


@pytest.mark.unit
class TestCurrentPhase:
    def test_channel_with_no_messages_is_never_exported(self, tmp_path: Path) -> None:
        assert plan(PHASE_CURRENT, None, tmp_path) == []
        assert plan(PHASE_BACKFILL, None, tmp_path) == []

    def test_nothing_posted_this_month_skips_the_current_month(self, tmp_path: Path) -> None:
        assert plan(PHASE_CURRENT, flake(2026, 10), tmp_path) == []

    def test_new_message_since_the_published_export_is_exported(self, tmp_path: Path) -> None:
        publish(tmp_path, CURRENT, [flake(2026, 11, 2)])
        assert plan(PHASE_CURRENT, flake(2026, 11, 18), tmp_path) == [CURRENT]

    def test_first_export_of_an_active_channel(self, tmp_path: Path) -> None:
        assert plan(PHASE_CURRENT, flake(2026, 11, 18), tmp_path) == [CURRENT]

    def test_published_export_already_holding_the_newest_message_is_skipped(
        self, tmp_path: Path
    ) -> None:
        newest = flake(2026, 11, 18)
        publish(tmp_path, CURRENT, [flake(2026, 11, 2), newest], NOW - timedelta(hours=2))
        assert plan(PHASE_CURRENT, newest, tmp_path) == []

    def test_unchanged_channel_is_refreshed_after_the_interval(self, tmp_path: Path) -> None:
        newest = flake(2026, 11, 18)
        publish(tmp_path, CURRENT, [newest], NOW - timedelta(hours=25))
        assert plan(PHASE_CURRENT, newest, tmp_path) == [CURRENT]
        assert plan(PHASE_CURRENT, newest, tmp_path, refresh_hours=48) == []

    def test_refresh_interval_of_zero_always_exports_active_channels(self, tmp_path: Path) -> None:
        newest = flake(2026, 11, 18)
        publish(tmp_path, CURRENT, [newest], NOW)
        assert plan(PHASE_CURRENT, newest, tmp_path, refresh_hours=0) == [CURRENT]
        # ...but still nothing for a channel with no messages this month.
        assert plan(PHASE_CURRENT, flake(2026, 10), tmp_path, refresh_hours=0) == []

    def test_deleted_newest_message_does_not_cause_endless_exports(self, tmp_path: Path) -> None:
        # The API keeps reporting a deleted message as the newest, so the
        # published export can never hold it. The cache records that the
        # channel was exported while that ID was the newest.
        newest = flake(2026, 11, 18)
        publish(tmp_path, CURRENT, [flake(2026, 11, 2)], NOW - timedelta(hours=1))
        entry = CacheEntry(newest, (NOW - timedelta(hours=1)).isoformat())
        assert plan(PHASE_CURRENT, newest, tmp_path, entry=entry) == []
        stale = CacheEntry(newest, (NOW - timedelta(hours=30)).isoformat())
        assert plan(PHASE_CURRENT, newest, tmp_path, entry=stale) == [CURRENT]

    def test_newest_message_from_a_clock_ahead_of_ours(self, tmp_path: Path) -> None:
        assert plan(PHASE_CURRENT, flake(2026, 12, 1, 0), tmp_path) == [CURRENT]


@pytest.mark.unit
class TestBackfillPhase:
    def test_no_month_after_the_newest_message_is_requested(self, tmp_path: Path) -> None:
        newest = flake(2026, 9, 10)
        publish(tmp_path, "2026-09", [newest])
        months = plan(PHASE_BACKFILL, newest, tmp_path, completed={"2026-09"})
        # Channel created 2026-08; 2026-10 and 2026-11 cannot hold messages.
        assert months == ["2026-08"]

    def test_known_empty_months_are_not_retried(self, tmp_path: Path) -> None:
        newest = flake(2026, 10, 10)
        publish(tmp_path, "2026-10", [newest])
        entry = CacheEntry(empty_months=["2026-08", "2026-09"])
        assert plan(PHASE_BACKFILL, newest, tmp_path, entry=entry, completed={"2026-10"}) == []

    def test_backfill_never_includes_the_current_month(self, tmp_path: Path) -> None:
        months = plan(PHASE_BACKFILL, flake(2026, 11, 18), tmp_path)
        assert months == ["2026-08", "2026-09", "2026-10"]

    def test_month_exported_before_its_last_messages_is_exported_again(
        self, tmp_path: Path
    ) -> None:
        # October looks complete, but its export predates the newest message.
        publish(tmp_path, "2026-10", [flake(2026, 10, 5)])
        done = {"2026-08", "2026-09", "2026-10"}
        assert plan(PHASE_BACKFILL, flake(2026, 10, 31, 23), tmp_path, completed=done) == [
            "2026-10"
        ]

    def test_complete_month_holding_the_newest_message_is_left_alone(self, tmp_path: Path) -> None:
        newest = flake(2026, 10, 31, 23)
        publish(tmp_path, "2026-10", [flake(2026, 10, 5), newest])
        done = {"2026-08", "2026-09", "2026-10"}
        assert plan(PHASE_BACKFILL, newest, tmp_path, completed=done) == []


@pytest.mark.unit
class TestPublishedHasMessage:
    def test_missing_or_broken_export_holds_nothing(self, tmp_path: Path) -> None:
        assert not published_has_message(tmp_path, CURRENT, "5")
        (tmp_path / CURRENT).mkdir()
        (tmp_path / CURRENT / f"{CURRENT}.json").write_text("{", encoding="utf-8")
        assert not published_has_message(tmp_path, CURRENT, "5")

    def test_a_later_message_proves_the_export_is_newer(self, tmp_path: Path) -> None:
        publish(tmp_path, CURRENT, ["3", "9"])
        assert published_has_message(tmp_path, CURRENT, "5")
        assert not published_has_message(tmp_path, CURRENT, "10")


@pytest.mark.unit
class TestExportCache:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "server" / "_export_state.json"
        cache = ExportCache(path)
        cache.load()
        cache.record_export("1", "900", NOW)
        cache.record_empty("1", "2026-09")
        cache.record_empty("1", "2026-08")
        cache.record_empty("1", "2026-09")
        cache.save()

        reloaded = ExportCache(path)
        reloaded.load()
        assert reloaded.get("1") == CacheEntry("900", NOW.isoformat(), ["2026-08", "2026-09"])
        assert reloaded.get("2") is None

    def test_unreadable_cache_is_an_empty_cache(self, tmp_path: Path) -> None:
        path = tmp_path / "_export_state.json"
        path.write_text("[1, 2", encoding="utf-8")
        cache = ExportCache(path)
        cache.load()
        assert cache.entries == {}

    def test_save_is_identical_when_nothing_changed(self, tmp_path: Path) -> None:
        path = tmp_path / "_export_state.json"
        cache = ExportCache(path)
        cache.record_export("2", "5", NOW)
        cache.record_export("1", "4", NOW)
        cache.save()
        first = path.read_bytes()
        again = ExportCache(path)
        again.load()
        again.save()
        assert path.read_bytes() == first


def _context(tmp_path: Path, activity: dict[str, ChannelActivity]) -> ChannelExportContext:
    cache = ExportCache(tmp_path / "public" / "srv" / "_export_state.json")
    cache.load()
    state: Any = type("State", (), {"update_channel": lambda *a: None})()
    return ChannelExportContext(
        config={"export": {"formats": ["html", "json"]}},
        token="t",
        server_key="srv",
        state_manager=state,
        activity=activity,
        cache=cache,
    )


def _process(
    context: ChannelExportContext, tmp_path: Path, phase: str, channel_id: str = CHANNEL_ID
) -> tuple[int, int, int, int, list[dict[str, str]]]:
    server_dir = tmp_path / "exports" / "srv"
    server_dir.mkdir(parents=True, exist_ok=True)
    return _process_single_channel(
        context,
        {"name": "general", "id": channel_id, "parent_id": None},
        ChannelType.REGULAR,
        server_dir,
        (["*"], []),
        {"general": "general"},
        tmp_path / "public",
        phase=phase,
    )


@pytest.mark.unit
class TestProcessSingleChannel:
    """The exporter runs DiscordChatExporter only for the months the plan names."""

    def test_unchanged_channel_runs_no_exports(self, tmp_path: Path) -> None:
        context = _context(tmp_path, {CHANNEL_ID: ChannelActivity(0, flake(2026, 9))})
        with (
            patch("scripts.export_channels.current_month_utc", return_value=CURRENT),
            patch("scripts.export_channels.run_export") as run,
        ):
            result = _process(context, tmp_path, PHASE_CURRENT)
        run.assert_not_called()
        assert result == (0, 1, 0, 0, [])
        assert context.unchanged == 1

    def test_changed_channel_is_exported_and_remembered(self, tmp_path: Path) -> None:
        newest = flake(2026, 11, 18)
        context = _context(tmp_path, {CHANNEL_ID: ChannelActivity(0, newest)})
        with (
            patch("scripts.export_channels.current_month_utc", return_value=CURRENT),
            patch("scripts.export_channels.run_export", return_value=(True, "")) as run,
        ):
            exports, updated, failed, _forbidden, errors = _process(
                context, tmp_path, PHASE_CURRENT
            )
            assert (exports, updated, failed, errors) == (2, 1, 0, [])
            assert run.call_count == 2  # noqa: PLR2004
            # The same run's record makes the next pass a no-op.
            run.reset_mock()
            assert _process(context, tmp_path, PHASE_CURRENT) == (0, 1, 0, 0, [])
            run.assert_not_called()
        assert context.cache is not None
        entry = context.cache.get(CHANNEL_ID)
        assert entry is not None and entry.last_message_id == newest

    def test_failed_export_is_not_remembered(self, tmp_path: Path) -> None:
        context = _context(tmp_path, {CHANNEL_ID: ChannelActivity(0, flake(2026, 11, 18))})
        with (
            patch("scripts.export_channels.current_month_utc", return_value=CURRENT),
            patch("scripts.export_channels.run_export", return_value=(False, "boom")),
        ):
            _exports, _updated, failed, _forbidden, _errors = _process(
                context, tmp_path, PHASE_CURRENT
            )
        assert failed == 2  # noqa: PLR2004
        assert context.cache is not None
        assert context.cache.get(CHANNEL_ID) is None

    def test_forum_is_skipped_without_running_the_exporter(self, tmp_path: Path) -> None:
        context = _context(tmp_path, {CHANNEL_ID: ChannelActivity(15, flake(2026, 11, 18))})
        with (
            patch("scripts.export_channels.current_month_utc", return_value=CURRENT),
            patch("scripts.export_channels.run_export") as run,
        ):
            for phase in (PHASE_CURRENT, PHASE_BACKFILL):
                assert _process(context, tmp_path, phase) == (0, 0, 0, 0, [])
        run.assert_not_called()

    def test_empty_past_month_is_recorded_and_not_retried(self, tmp_path: Path) -> None:
        # Created 2026-08, newest message 2026-09: backfill asks for both
        # months. The exporter returns an empty August.
        newest = flake(2026, 9, 10)
        context = _context(tmp_path, {CHANNEL_ID: ChannelActivity(0, newest)})

        def fake_export(cmd: list[str], timeout: int = 900) -> tuple[bool, str]:
            output = Path(cmd[cmd.index("-o") + 1])
            if output.suffix == ".json":
                ids = [] if output.stem == "2026-08" else [newest]
                output.write_text(json.dumps({"messages": [{"id": i} for i in ids]}))
            return True, ""

        with (
            patch("scripts.export_channels.current_month_utc", return_value=CURRENT),
            patch("scripts.export_channels.run_export", side_effect=fake_export) as run,
        ):
            _process(context, tmp_path, PHASE_BACKFILL)
            first_calls = run.call_count
            # September is not published yet (organize has not run in this
            # test), so only the cache can stop a second round.
            run.reset_mock()
            _process(context, tmp_path, PHASE_BACKFILL)
            second_months = {
                Path(c.args[0][c.args[0].index("-o") + 1]).stem for c in run.mock_calls
            }
        assert first_calls == 4  # noqa: PLR2004  # two months, two formats
        assert context.cache is not None
        entry = context.cache.get(CHANNEL_ID)
        assert entry is not None
        assert entry.empty_months == ["2026-08"]
        assert entry.last_message_id == newest
        assert "2026-08" not in second_months

    def test_channel_without_activity_facts_is_exported_as_before(self, tmp_path: Path) -> None:
        context = _context(tmp_path, {})
        with (
            patch("scripts.export_channels.current_month_utc", return_value=CURRENT),
            patch("scripts.export_channels.run_export", return_value=(True, "")) as run,
        ):
            exports, updated, _failed, _forbidden, _errors = _process(
                context, tmp_path, PHASE_CURRENT
            )
        assert (exports, updated) == (2, 1)
        assert run.call_count == 2  # noqa: PLR2004
        assert context.unchanged == 0


@pytest.mark.unit
class TestActivityLookupFailure:
    def test_failure_is_reported_and_every_channel_is_still_exported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = {
            "site": {},
            "servers": {
                "srv": {
                    "name": "Srv",
                    "guild_id": "1",
                    "include_channels": ["*"],
                    "exclude_channels": [],
                }
            },
            "export": {"formats": ["json"], "include_threads": "all"},
        }
        channels = [{"name": "general", "id": CHANNEL_ID, "parent_id": None}]
        with (
            patch("scripts.export_channels.load_config", return_value=config),
            patch("scripts.export_channels.get_bot_token", return_value="t"),
            patch(
                "scripts.export_channels.fetch_guild_channels",
                return_value=(channels, {"general": "general"}),
            ),
            patch(
                "scripts.export_channels.fetch_guild_activity",
                side_effect=DiscordApiError(401, "bad token"),
            ),
            patch("scripts.export_channels.current_month_utc", return_value=CURRENT),
            patch("scripts.export_channels.run_export", return_value=(True, "")) as run,
        ):
            summary = export_all_channels(public_dir=tmp_path / "public")
        assert run.called
        assert summary["channels_unchanged"] == 0
        assert len(summary["errors"]) == 1
        assert "Failed to fetch channel activity" in summary["errors"][0]["error"]
