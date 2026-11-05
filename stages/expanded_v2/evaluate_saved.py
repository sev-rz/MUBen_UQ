"""Evaluate frozen fits with a shape-aware loader for the upstream SWAG state.

The upstream constructor creates rank-one covariance buffers, whereas a fitted
state has twenty rows. Resize only those registered buffers before strict load.
This operational fix changes neither posterior values nor frozen experiment
settings. Original run_expansion.py remains unmodified and hash-verifiable.
"""
import torch
import run_expansion as experiment
original_class=experiment.swag_class

def reloadable_swag_class():
    Base=original_class()
    class ReloadableSWAG(Base):
        def load_state_dict(self,state_dict,strict=True,assign=False):
            for name,value in state_dict.items():
                if name.endswith('_CovMatSqrt'):
                    assert name in self._buffers and value.ndim==2
                    self._buffers[name]=torch.empty_like(value)
            result=super().load_state_dict(state_dict,strict=strict,assign=assign)
            for name,value in self.state_dict().items():
                torch.testing.assert_close(value,state_dict[name],rtol=0,atol=0)
            return result
    return ReloadableSWAG

if __name__=='__main__':
    experiment.swag_class=reloadable_swag_class
    experiment.evaluate_all()
