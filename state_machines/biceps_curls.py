import time

from .base_state_machine import BaseStateMachine
class BicepsCurls(BaseStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.START_ANGLE = 160.0       # Fully extended arm
        self.PEAK_ANGLE = 70.0         # Fully flexed arm
        self.VELOCITY_THRESHOLD = 15.0 # Degrees per second to consider "moving"
        self.HOLD_TIME_MIN = 0.5       # Seconds required to register a hold

    def _get_smoothed_velocity(self, delta):
        return super()._get_smoothed_velocity(delta)

    def update(self, current_angle, delta_angle, angular_velocity):
        super().update(current_angle, delta_angle, angular_velocity)
        velocity = self._get_smoothed_velocity(angular_velocity)

        if self.current_state == self.STATE_READY:
            if velocity < -self.VELOCITY_THRESHOLD: 
                self.current_state = self.STATE_MOVING_UP

        elif self.current_state == self.STATE_MOVING_UP:
            if (
                abs(velocity) < self.VELOCITY_THRESHOLD
                and current_angle <= (self.PEAK_ANGLE + 20)
            ):
                self.current_state = self.STATE_HOLDING
                self.hold_start_time = time.time()

            elif (
                velocity > self.VELOCITY_THRESHOLD
                and current_angle <= (self.PEAK_ANGLE + 20)
            ):
                self.current_state = self.STATE_MOVING_DOWN

        elif self.current_state == self.STATE_HOLDING:
            if velocity > self.VELOCITY_THRESHOLD:
                self.current_state = self.STATE_MOVING_DOWN

        elif self.current_state == self.STATE_MOVING_DOWN:
            # Complete the rep when the arm returns near its extended position.
            # The velocity check is intentionally omitted because the arm can
            # still be moving when it crosses the completion angle.
            if current_angle >= (self.START_ANGLE - 20):
                self.current_state = self.STATE_REP_COMPLETE

        elif self.current_state == self.STATE_REP_COMPLETE:
            self.rep_count += 1
            print(f"🥇 Rep {self.rep_count} Complete!")
            self.current_state = self.STATE_READY

        return self
