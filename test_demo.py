import unittest
import numpy as np
from demo import basis, unit_acceleration, fit_physics, fit_ar, forecast_ar


class DemonstrationTests(unittest.TestCase):
    def test_ode_against_exact_solution(self):
        t = np.arange(500)/100
        np.testing.assert_allclose(unit_acceleration(t, 6., .03), basis(t, 6., .03), atol=1e-6)

    def test_noiseless_parameter_recovery(self):
        t = np.arange(400)/100
        np.testing.assert_allclose(fit_physics(t, .04*basis(t, 6., .03)), [6., .03, .04], atol=1e-6)

    def test_orientation_amplitude_nonidentifiability(self):
        t = np.arange(100)/100
        beta = .5*.1*np.cos(np.pi/6)
        alternative_amplitude = beta/(.5*np.cos(np.pi/3))
        np.testing.assert_allclose(.5*.1*np.cos(np.pi/6)*basis(t, 6., .03),
                                   .5*alternative_amplitude*np.cos(np.pi/3)*basis(t, 6., .03))

    def test_ar_recovers_exact_recurrence(self):
        y = np.cos(np.arange(400)*.15)
        c = fit_ar(y[:300], order=2)
        np.testing.assert_allclose(forecast_ar(y[:300], c, 100), y[300:], atol=1e-9)

    def test_forecast_does_not_mutate_history(self):
        y = np.cos(np.arange(100)*.15)
        saved = y.copy()
        forecast_ar(y, fit_ar(y, 2), 50)
        np.testing.assert_array_equal(y, saved)


if __name__ == '__main__':
    unittest.main()
