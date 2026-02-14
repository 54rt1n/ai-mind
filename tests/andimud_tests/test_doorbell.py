"""Unit tests for Doorbell target delivery behavior."""

from unittest.mock import MagicMock


def test_ring_character_target_is_private():
    """If character targets are resolved, only those characters hear the ring."""
    from typeclasses.doorbell import Doorbell

    bell = MagicMock()
    bell.key = "Front Door Doorbell"
    bell.dbref = "#500"
    bell.db = MagicMock()
    bell.db.include_location = True
    bell._resolve_targets = MagicMock()
    bell._resolve_targets.return_value = ([], [MagicMock(), MagicMock()])
    bell._get_ring_message = MagicMock(return_value="You hear the bell ring.")
    bell._publish_notification = MagicMock()
    bell._publish_notification_to_character = MagicMock()

    caller = MagicMock()
    caller.msg = MagicMock()

    Doorbell.ring(bell, caller=caller)

    targets = bell._resolve_targets.return_value[1]
    for target in targets:
        target.msg.assert_called_once_with("You hear the bell ring.")

    bell._publish_notification.assert_not_called()
    assert bell._publish_notification_to_character.call_count == len(targets)
    caller.msg.assert_called_once_with("You ring the Front Door Doorbell.")


def test_ring_room_broadcast_when_no_character_target():
    """Room broadcasting remains unchanged when no character target is set."""
    from typeclasses.doorbell import Doorbell

    origin_room = MagicMock()
    origin_room.is_typeclass.return_value = True
    origin_room.msg_contents = MagicMock()

    bell = MagicMock()
    bell.key = "Front Door Doorbell"
    bell.dbref = "#500"
    bell.db = MagicMock()
    bell._targeted_turn_enabled = MagicMock(return_value=True)
    bell.db.include_location = True
    bell.location = origin_room
    bell._resolve_targets = MagicMock(return_value=([], []))
    bell._get_local_message = MagicMock(return_value="local ring")
    bell._get_ring_message = MagicMock(return_value="remote ring")
    bell._publish_notification = MagicMock()

    caller = MagicMock()
    caller.msg = MagicMock()

    Doorbell.ring(bell, caller=caller)

    origin_room.msg_contents.assert_called_once_with("local ring")
    bell._publish_notification.assert_called_once()
    caller.msg.assert_called_once_with("You ring the Front Door Doorbell.")


def test_ring_character_target_uses_message_override():
    """Custom message override should be delivered to targeted characters."""
    from typeclasses.doorbell import Doorbell

    bell = MagicMock()
    bell.key = "Front Door Doorbell"
    bell.dbref = "#500"
    bell.db = MagicMock()
    bell.db.include_location = True
    bell._resolve_targets = MagicMock()
    bell._resolve_targets.return_value = ([], [MagicMock()])
    bell._get_ring_message = MagicMock(return_value="default ring")
    bell._publish_notification_to_character = MagicMock()

    caller = MagicMock()
    caller.msg = MagicMock()

    Doorbell.ring(bell, caller=caller, message_override="custom ring text")

    target = bell._resolve_targets.return_value[1][0]
    target.msg.assert_called_once_with("custom ring text")
    bell._publish_notification_to_character.assert_called_once_with(
        target, "custom ring text"
    )


def test_publish_notification_to_character_sets_assigns_turn_when_enabled():
    """Targeted events request turns when targeted_turn is enabled."""
    from unittest.mock import patch
    from typeclasses.doorbell import Doorbell

    bell = MagicMock()
    bell.key = "Front Door Doorbell"
    bell.dbref = "#500"
    bell.db = MagicMock()
    bell._targeted_turn_enabled = MagicMock(return_value=True)

    target = MagicMock()
    target.is_typeclass.return_value = True
    target.key = "Andi"
    target.dbref = "#42"
    target.db = MagicMock()
    target.db.agent_id = "andi"
    target.location = None

    with patch("typeclasses.doorbell.redis.from_url") as mock_redis, patch(
        "typeclasses.doorbell.SyncRedisMUDClient"
    ) as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_redis.return_value = MagicMock()

        Doorbell._publish_notification_to_character(bell, target, "ring")

        kwargs = mock_client.publish_mud_event.call_args.kwargs
        metadata = kwargs["metadata"]
        assert metadata["assigns_turn"] is True
        assert "notification_type" not in metadata


def test_publish_notification_to_character_disables_assigns_turn_when_disabled():
    """Targeted events skip turn assignment when targeted_turn is disabled."""
    from unittest.mock import patch
    from typeclasses.doorbell import Doorbell

    bell = MagicMock()
    bell.key = "Front Door Doorbell"
    bell.dbref = "#500"
    bell.db = MagicMock()
    bell._targeted_turn_enabled = MagicMock(return_value=False)

    target = MagicMock()
    target.is_typeclass.return_value = True
    target.key = "Andi"
    target.dbref = "#42"
    target.db = MagicMock()
    target.db.agent_id = "andi"
    target.location = None

    with patch("typeclasses.doorbell.redis.from_url") as mock_redis, patch(
        "typeclasses.doorbell.SyncRedisMUDClient"
    ) as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_redis.return_value = MagicMock()

        Doorbell._publish_notification_to_character(bell, target, "ring")

        kwargs = mock_client.publish_mud_event.call_args.kwargs
        metadata = kwargs["metadata"]
        assert metadata["assigns_turn"] is False


def test_publish_notification_to_character_falls_back_to_key_when_agent_id_missing():
    """If db.agent_id is missing, fallback to normalized character key."""
    from unittest.mock import patch
    from typeclasses.doorbell import Doorbell

    bell = MagicMock()
    bell.key = "Front Door Doorbell"
    bell.dbref = "#500"
    bell.db = MagicMock()
    bell._targeted_turn_enabled = MagicMock(return_value=True)

    target = MagicMock()
    target.is_typeclass.return_value = True
    target.key = "Andi"
    target.dbref = "#42"
    target.db = MagicMock()
    target.db.agent_id = None
    target.location = None

    with patch("typeclasses.doorbell.redis.from_url") as mock_redis, patch(
        "typeclasses.doorbell.SyncRedisMUDClient"
    ) as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_redis.return_value = MagicMock()

        Doorbell._publish_notification_to_character(bell, target, "ring")

        kwargs = mock_client.publish_mud_event.call_args.kwargs
        metadata = kwargs["metadata"]
        assert metadata["target_agent_id"] == "andi"
        assert metadata["assigns_turn"] is True


def test_targeted_turn_enabled_parses_false_string():
    """String values from @set should parse correctly for toggles."""
    from typeclasses.doorbell import Doorbell

    bell = MagicMock()
    bell.db = MagicMock()
    bell.db.targeted_turn = "False"

    assert Doorbell._targeted_turn_enabled(bell) is False


def test_sync_doorbell_cmdset_removes_duplicates_before_add():
    """Doorbell cmdset sync should remove duplicates and add one fresh cmdset."""
    from typeclasses.doorbell import Doorbell, DOORBELL_CMDSET

    bell = MagicMock()
    bell.cmdset = MagicMock()
    bell.cmdset.has.side_effect = [True, True, False]

    Doorbell._sync_doorbell_cmdset(bell)

    assert bell.cmdset.delete.call_count == 2
    bell.cmdset.add.assert_called_once_with(DOORBELL_CMDSET, persistent=True)


def test_sync_doorbell_cmdset_adds_when_missing():
    """Doorbell cmdset sync should add cmdset when none is present."""
    from typeclasses.doorbell import Doorbell, DOORBELL_CMDSET

    bell = MagicMock()
    bell.cmdset = MagicMock()
    bell.cmdset.has.return_value = False

    Doorbell._sync_doorbell_cmdset(bell)

    bell.cmdset.delete.assert_not_called()
    bell.cmdset.add.assert_called_once_with(DOORBELL_CMDSET, persistent=True)
