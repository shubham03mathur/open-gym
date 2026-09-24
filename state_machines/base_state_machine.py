class BaseStateMachine:
    def __init__(self) -> None:
        self.STATE_READY = "READY"
        self.STATE_MOVING_UP = "MOVING_UP"
        self.STATE_HOLDING = "HOLDING"
        self.STATE_MOVING_DOWN = "MOVING_DOWN"
        self.STATE_REP_COMPLETE = "REP_COMPLETE"
        self.current_state = self.STATE_READY

        self.hold_start_time = None
        self.rep_count = 0
        self.velocity_history = []     # For smoothing noise

    def _get_smoothed_velocity(self, raw_velocity):
        """Applies a simple moving average to filter out camera jitter."""
        self.velocity_history.append(raw_velocity)
        if len(self.velocity_history) > 3:  # Smooth over last 3 frames
            self.velocity_history.pop(0)
        return sum(self.velocity_history) / len(self.velocity_history)

    def update(self, current_angle, delta_angle, delta_t):
        pass