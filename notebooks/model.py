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
