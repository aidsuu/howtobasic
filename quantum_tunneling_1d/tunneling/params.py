from dataclasses import dataclass

@dataclass
class SimParams:
    # Domain
    x_min: float = -200.0
    x_max: float = 200.0
    nx: int = 2000
    
    # Time
    dt: float = 0.02
    n_steps: int = 5000
    snapshot_every: int = 20
    
    # Physical constansts, atomic-like units by default
    hbar: float = 1.0
    mass: float = 1.0
    
    # Barrier
    barrier_left: float = -5.0
    barrier_right: float = 5.0
    barrier_height: float = 1.5
    
    # Initial wave packet
    x0: float = -80.0
    sigma: float = 8.0
    k0: float = 1.2
    
    