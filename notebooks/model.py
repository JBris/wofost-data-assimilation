import pypfilt
import scipy

class GaussianWalk(pypfilt.Model):
    def field_types(self, ctx):
        import numpy as np

        return [('x', np.dtype(float))]

    def update(self, ctx, time_step, is_fs, prev, curr):
        """Perform a single time-step."""
        rnd = ctx.component['random']['model']
        step = rnd.normal(loc=0, scale=0.01, size=curr.shape)
        curr['x'] = prev['x'] + step

    def can_smooth(self):
        return { 'x' }
    
class GaussianObs(pypfilt.obs.Univariate):
    def distribution(self, ctx, snapshot):
        import scipy
        expected = snapshot.state_vec[self.unit]
        sdev = self.settings['parameters']['sdev']
        return scipy.stats.norm(loc=expected, scale=sdev)

class WofostObs(pypfilt.obs.Univariate):
    def distribution(self, ctx, snapshot):
        import scipy
        expected = snapshot.state_vec[self.unit]
        sdev = self.settings['parameters']['sdev'] * expected
        sdev[sdev == 0] = 1e-10           
        return scipy.stats.norm(loc=expected, scale=sdev)

class Wofost(pypfilt.Model):
    def field_types(self, ctx):
        """
        Define the state vector structure.
        """
        import numpy as np
        return [
            ('TDWI', np.float64),
            ('WAV', np.float64),
            ('SPAN', np.float64),
            ('SMFCF', np.float64),
            ('LAI', np.float64),
            ('SM', np.float64),
        ]

    def can_smooth(self):
        """
        The fields that can be smoothed by the post-regularisation filter.
        """
        return { 'LAI', 'SM' }

    def init(self, ctx, vec):
        """
        Initialise the state vectors.
        """
        import copy
        from dataproviders import parameters, agromanagement, weather
        from pcse.models import Wofost72_WLP_FD

        particles = ctx.settings["filter"]["particles"]
        self.ensemble = []
        prior = ctx.data['prior']
        outputs = list(ctx.settings["observations"].keys())

        for i in range(particles):
            p = copy.deepcopy(parameters)
            for par, distr in prior.items():
                if par in outputs:
                    continue
                p.set_override(par, distr[i])
            member = Wofost72_WLP_FD(p, weather, agromanagement)
            self.ensemble.append(member)

    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        Update the state vectors.
        """
        outputs = list(ctx.settings["observations"].keys())
        for i, member in enumerate(self.ensemble):
            member.run(1)
            for output_name in outputs:
                output = member.get_variable(output_name)
                curr[output_name][i] = output
 