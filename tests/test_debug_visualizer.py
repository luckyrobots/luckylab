"""Tests for DebugVisualizer throttle and drawing primitives."""

import time

from luckylab.viewer.debug_draw import DebugVisualizer


class MockClient:
    """Fake gRPC client that records draw calls."""

    def __init__(self):
        self.arrows: list[dict] = []
        self.lines: list[dict] = []

    def draw_arrow(self, *, origin, direction, color, scale, clear_previous):
        self.arrows.append(
            {
                "origin": origin,
                "direction": direction,
                "color": color,
                "scale": scale,
                "clear_previous": clear_previous,
            }
        )
        return True

    def draw_line(self, *, start, end, color, clear_previous):
        self.lines.append(
            {
                "start": start,
                "end": end,
                "color": color,
                "clear_previous": clear_previous,
            }
        )
        return True


class MockLuckyRobots:
    def __init__(self, client):
        self.engine_client = client


class MockEnv:
    def __init__(self, client=None):
        if client is not None:
            self.luckyrobots = MockLuckyRobots(client)


class TestDebugVisualizerThrottle:
    """Tests for the should_draw() throttle mechanism."""

    def test_should_draw_true_on_first_call(self):
        viz = DebugVisualizer(MockEnv(MockClient()))
        assert viz.should_draw() is True

    def test_should_draw_false_within_interval(self):
        viz = DebugVisualizer(MockEnv(MockClient()), draw_interval_ms=1000.0)
        assert viz.should_draw() is True
        assert viz.should_draw() is False

    def test_should_draw_true_after_interval(self):
        viz = DebugVisualizer(MockEnv(MockClient()), draw_interval_ms=10.0)
        assert viz.should_draw() is True
        time.sleep(0.015)  # 15ms > 10ms interval
        assert viz.should_draw() is True


class TestDebugVisualizerDrawArrow:
    """Tests for draw_arrow primitive."""

    def test_draw_arrow_calls_client(self):
        client = MockClient()
        viz = DebugVisualizer(MockEnv(client))
        result = viz.draw_arrow(
            origin=(0.0, 0.0, 0.5),
            direction=(1.0, 0.0, 0.0),
            color=(1.0, 0.0, 0.0, 1.0),
        )
        assert result is True
        assert len(client.arrows) == 1
        assert client.arrows[0]["origin"] == (0.0, 0.0, 0.5)
        assert client.arrows[0]["direction"] == (1.0, 0.0, 0.0)

    def test_draw_arrow_passes_clear_previous(self):
        client = MockClient()
        viz = DebugVisualizer(MockEnv(client))
        viz.draw_arrow(
            origin=(0, 0, 0),
            direction=(1, 0, 0),
            color=(1, 0, 0, 1),
            clear_previous=True,
        )
        assert client.arrows[0]["clear_previous"] is True

    def test_draw_arrow_returns_false_without_client(self):
        viz = DebugVisualizer(MockEnv())  # no client
        result = viz.draw_arrow(
            origin=(0, 0, 0),
            direction=(1, 0, 0),
            color=(1, 0, 0, 1),
        )
        assert result is False


class TestDebugVisualizerDrawLine:
    """Tests for draw_line primitive."""

    def test_draw_line_calls_client(self):
        client = MockClient()
        viz = DebugVisualizer(MockEnv(client))
        result = viz.draw_line(
            start=(0.0, 0.0, 0.0),
            end=(1.0, 1.0, 1.0),
            color=(1.0, 1.0, 1.0, 1.0),
        )
        assert result is True
        assert len(client.lines) == 1

    def test_draw_line_returns_false_without_client(self):
        viz = DebugVisualizer(MockEnv())
        result = viz.draw_line(
            start=(0, 0, 0),
            end=(1, 1, 1),
            color=(1, 1, 1, 1),
        )
        assert result is False


class TestDebugVisualizerClientResolution:
    """Tests for client property resolution."""

    def test_client_from_luckyrobots(self):
        client = MockClient()
        viz = DebugVisualizer(MockEnv(client))
        assert viz.client is client

    def test_client_none_when_no_luckyrobots(self):
        viz = DebugVisualizer(MockEnv())
        assert viz.client is None

    def test_warns_only_once_when_no_client(self, capsys):
        viz = DebugVisualizer(MockEnv())
        viz.draw_arrow(origin=(0, 0, 0), direction=(1, 0, 0), color=(1, 0, 0, 1))
        viz.draw_arrow(origin=(0, 0, 0), direction=(1, 0, 0), color=(1, 0, 0, 1))
        captured = capsys.readouterr()
        assert captured.out.count("No engine client") == 1
