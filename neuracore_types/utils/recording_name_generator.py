"""Generate memorable, lightly technical two-word recording codenames."""

import random

# fmt: off
# cspell:disable

RECORDING_DESCRIPTORS = [
    # Mathematics and algorithms
    "Adaptive", "Affine", "Algebraic", "Algorithmic", "Analytic", "Atomic",
    "Axiomatic", "Binary", "Canonical", "Cybernetic", "Differential",
    "Dynamic", "Eigen", "Euclidean", "Fourier", "Fractal", "Gaussian",
    "Geometric", "Gradient", "Harmonic", "Helical", "Isometric", "Iterative",
    "Kinematic", "Kinetic", "Logical", "Manifold", "Metric", "Modular",
    "Orbital", "Orthogonal", "Parallel", "Parametric", "Quantum", "Radial",
    "Recursive", "Rotational", "Spectral", "Symmetric", "Topological",
    "Variational", "Vectorial", "Wavelet",
    "Abelian", "Arithmetic", "Bayesian", "Cartesian", "Lagrangian",
    "Laplacian", "Tensorial",

    # Robotics and control
    "Actuated", "Articulated", "Autonomous", "Calibrated", "ClosedLoop",
    "Compliant", "Coordinated", "Decentralized", "Deterministic",
    "Distributed", "Embodied", "Haptic", "Holonomic", "Inertial",
    "Mechatronic", "Multimodal", "Neural", "Perceptive", "Predictive",
    "Reactive", "Redundant", "Resilient", "Robotic", "Sensorial", "Servoed",
    "Synchronized", "Tactile", "Teleoperated", "Temporal", "Trajectory",
    "Feedback", "Feedforward", "Stabilized", "Regulated", "Controlled",
    "Observable", "Controllable", "Maneuverable", "Dexterous", "Articulatory",
    "Locomotive", "Manipulative", "Navigational", "Proprioceptive",
    "Exteroceptive", "Collaborative", "Cognitive", "Behavioral", "Emergent",
    "TaskAware",

    # Mathematics and statistics
    "Angular", "Asymptotic", "Bilinear", "Bounded", "Chaotic",
    "Combinatorial", "Convex", "Discrete", "Elliptic", "Exponential",
    "Finite", "Functional", "Homogeneous", "Hyperbolic", "Integral", "Linear",
    "Logarithmic", "Markovian", "Monotonic", "Nonlinear", "Numerical",
    "Polynomial", "Probabilistic", "Scalar", "Stochastic", "Continuous",
    "Differentiable", "Dimensional", "Factorial", "Fractional",
    "Infinitesimal", "Injective", "Surjective", "Bijective", "Periodic",
    "Piecewise", "Quadratic", "Rational", "Real", "Complex", "Prime",
    "Ordinal", "Cardinal", "Statistical", "Distributive", "Associative",
    "Commutative", "Transitive", "Normalized", "Ergodic",

    # Engineering and physics
    "Acoustic", "Aerodynamic", "Analog", "Automated", "Capacitive", "Coherent",
    "Conductive", "Digital", "Electromagnetic", "Electronic", "Embedded",
    "Energetic", "Hydraulic", "Mechanical", "Optical", "Pneumatic",
    "Programmable", "Resonant", "Thermal", "Wireless", "Electrical",
    "Magnetic", "Photonic", "Sonic", "Ultrasonic", "Vibrational",
    "Volumetric", "Thermodynamic", "Aerostatic", "Hydrostatic",
    "Electromechanical", "Electrostatic", "Magnetostatic", "Piezoelectric",
    "Tribological", "Structural", "Material", "Metallic", "Ceramic",
    "Composite", "Deformable", "Rigid", "Flexible", "Compressible",
    "Incompressible", "Viscous", "Turbulent", "Laminar", "Ballistic",
    "Gyroscopic",

    # Computing and data
    "Asynchronous", "Buffered", "Cached", "Compiled", "Composable",
    "Concurrent", "Configurable", "Connected", "Containerized", "DataDriven",
    "Declarative", "Encoded", "Federated", "Indexed", "Inferred",
    "Interpreted", "Latent", "Learned", "MemorySafe", "Networked",
    "Optimized", "Pipelined", "Programmatic", "Quantized", "RealTime",
    "Scalable", "Stateful", "Stateless", "Streaming", "Symbolic", "Threaded",
    "Tokenized", "Typed", "Virtual", "Immutable", "Idempotent", "Replicated",
    "Sharded", "Sparse", "Dense", "Secure", "Encrypted", "FaultTolerant",
    "EventDriven", "ModelBased", "SimulationReady", "HardwareAware",
    "SoftwareDefined", "EdgeAware", "ComputeBound",

    # Space and geometry
    "Axial", "Central", "Circular", "Compact", "Concentric", "Coplanar",
    "Cylindrical", "Diagonal", "Directional", "Equidistant", "Horizontal",
    "Lateral", "Longitudinal", "Planar", "Polar", "Proximal", "Rectilinear",
    "Spherical", "Tangential", "Translational", "Transverse", "Vertical",
    "Local", "Global", "Nested", "Layered", "Adjacent", "Aligned", "Centered",
    "Offset", "Mirrored", "Inverted", "Upright", "Oblique", "Peripheral",
    "Interior", "Exterior", "Upstream", "Downstream", "Forward", "Reverse",
    "Leftward", "Rightward", "Clockwise", "Counterclockwise", "Multiaxial",
    "Omnidirectional", "Bidirectional", "Unidirectional", "Spatial",

    # Materials, light, and space
    "Amber", "Azure", "Cobalt", "Copper", "Crimson", "Golden", "Graphite",
    "Indigo", "Ivory", "Jade", "Obsidian", "Onyx", "Quartz", "Ruby",
    "Sapphire", "Silver", "Titanium", "Umber", "Vermilion", "Violet",
    "Astral", "Celestial", "Cosmic", "Galactic", "Lunar", "Martian",
    "Nebular", "Solar", "Stellar", "Interstellar", "Chromatic", "Iridescent",
    "Luminous", "Radiant", "Translucent", "Transparent", "Opaque",
    "Reflective", "Refractive", "Prismatic", "Fluorescent", "Phosphorescent",
    "Monochrome", "Polychrome", "Ultraviolet", "Infrared", "Visible",
    "Luminal", "Nocturnal", "Diurnal",

    # Performance and reliability
    "Accurate", "Active", "Advanced", "Agile", "Alert", "Balanced", "Capable",
    "Clear", "Consistent", "Crisp", "Decisive", "Diligent", "Efficient",
    "Exact", "Focused", "Graceful", "Precise", "Quick", "Rapid", "Ready",
    "Reliable", "Responsive", "Robust", "Sharp", "Smooth", "Stable",
    "Steady", "Swift", "Systematic", "Versatile", "Vigilant", "Faultless",
    "Durable", "Dependable", "Repeatable", "Reproducible", "Maintainable",
    "Portable", "Practical", "Economical", "Optimal", "Superior", "Refined",
    "Polished", "Seamless", "Timely", "Orderly", "Methodical", "Nimble",
    "Performant",

    # Perception, signals, and representation
    "Perceptual", "Semantic", "Photometric", "Geodetic", "Attentive",
    "Salient", "Inferential", "Evidential", "Projective", "Topometric",
    "Egocentric", "Relational", "Contextual", "Conceptual", "Informational",
    "Entropic", "Holographic", "Interferometric", "Diffractive", "Radiometric",
    "Wavefront", "Cepstral", "Topographic", "Geodesic", "ScaleInvariant",
    "RotationInvariant", "TranslationInvariant", "Equivariant", "Invariant",
    "Covariant", "Contravariant", "Isotropic", "Anisotropic", "Homomorphic",
    "Isomorphic", "Homeomorphic", "Hermitian", "Riemannian", "Symplectic",
    "Unitary", "Multilinear", "Nonparametric", "Polyhedral", "Curvilinear",
    "Barycentric", "Biquadratic", "Conformal", "Aperiodic", "Quasiperiodic",
    "Vortical",

    # Planning, learning, and dynamic systems
    "Causal", "Counterfactual", "Generative", "Discriminative", "Imitative",
    "Contrastive", "Transferable", "Generalizable", "Compositional",
    "Hierarchical", "LatentSpace", "RecedingHorizon", "ConsensusBased",
    "GraphBased", "SamplingBased", "SearchGuided", "OptimizationDriven",
    "ConstraintAware", "Feasible", "Reachable", "Cooperative", "Convergent",
    "Divergent", "Dissipative", "Conservative", "Bistable", "Multistable",
    "Metastable", "Damped", "Undamped", "Oscillatory", "PhaseLocked",
    "FrequencyLocked", "TimeInvariant", "TimeVarying", "Hysteretic",
    "Saturated", "Linearized", "Decoupled", "Coupled", "Interconnected",
    "SelfOrganizing", "SelfConsistent", "SelfSimilar", "Lyapunov", "Riccati",
    "Hamiltonian", "Lipschitz", "Monoidal", "Functorial",
]

RECORDING_CODENAMES = [
    # Astronomical codenames
    "Aldebaran", "Alcor", "Alcyone", "Alhena", "Alnair", "Alnilam",
    "Alphard", "Alsephina", "Altair", "Amalthea", "Ananke", "Andromeda",
    "Antares", "Aphelion", "Aquila", "Arcturus", "Ariel", "Aurora", "Avior",
    "Bellatrix", "Betelgeuse", "Borealis", "Callisto", "Canopus", "Capella",
    "Carina", "Cassiopeia", "Celaeno", "Centauri", "Ceres", "Charon",
    "Columba", "Comet", "Cosmos", "Crux", "Cygnus", "Deneb", "Despina",
    "Dione", "Draco", "Eclipse", "Elara", "Electra", "Enceladus", "Equinox",
    "Eridanus", "Ersa", "Europa", "Fornax", "Ganymede", "Grus", "Halley",
    "Helios", "Himalia", "Hyperion", "Hydra", "Iapetus", "Io", "Jupiter",
    "Lacerta", "Larissa", "Leda", "Luna", "Lyra", "Maia", "Mars", "Megrez",
    "Menkar", "Mercury", "Meteor", "Meridian", "Metis", "Mimas", "Miranda",
    "Naos", "Nashira", "Nebula", "Neptune", "Nereid", "Nova", "Oberon",
    "Octans", "Ophelia", "Orion", "Parallax", "Pavo", "Pegasus",
    "Perihelion", "Phobos", "Phoebe", "Polaris", "Pollux", "Procyon",
    "Proteus", "Pulsar", "Pyxis", "Quasar", "Regulus", "Rhea", "Rigel",
    "Sabik", "Saiph", "Saturn", "Sedna", "Sirius", "Skoll", "Sol",
    "Solstice", "Spica", "Talitha", "Tarvos", "Telesto", "Thalassa", "Thebe",
    "Themis", "Thuban", "Titan", "Titania", "Triton", "Tucana", "Umbriel",
    "Vega", "Vela", "Venus", "Volans", "Wasat", "Ymir", "Zaniah", "Zenith",
    "Zosma",

    # Selected mathematics, computing, science, and robotics pioneers
    "Abel", "Alhazen", "Archimedes", "Babbage", "Bayes", "Bernoulli",
    "Boltzmann", "Braitenberg", "Cantor", "Capek", "Church", "Copernicus",
    "Curie", "Descartes", "Devol", "Dijkstra", "Dirac", "Engelberger",
    "Euclid", "Faraday", "Feigenbaum", "Feynman", "Franklin",
    "Galois", "Galileo", "Hilbert", "Hopper", "Hubble", "Jemison", "Kepler",
    "Knuth", "Lamarr", "Lamport", "Lavoisier", "Leibniz", "Lovelace",
    "Marconi", "Maxwell", "McCarthy", "McClintock", "Meitner", "Minsky",
    "Moravec", "Neumann", "Newell", "Nilsson", "Noether", "Pascal",
    "Planck", "Poincare", "Raman", "Ritchie", "Rosenblatt", "Sagan",
    "Shannon", "Simon", "Turing", "Vaughan", "Zuse",

    # Additional science, mathematics, and computing pioneers
    "Albattani", "Allen", "Almeida", "Antonelli", "Agnesi", "Ardinghelli",
    "Aryabhata", "Austin", "Banach", "Banzai", "Bardeen", "Bartik",
    "Bassi", "Beaver", "Bell", "Benz", "Bhabha", "Bhaskara",
    "Black", "Blackburn", "Blackwell", "Bohr", "Booth", "Borg",
    "Bose", "Bouman", "Boyd", "Brahmagupta", "Brattain", "Brown",
    "Buck", "Burnell", "Cannon", "Carson", "Cartwright", "Carver",
    "Cerf", "Chandrasekhar", "Chaplygin", "Chatelet", "Chatterjee",
    "Chebyshev", "Cohen", "Chaum", "Clarke", "Colden", "Cori", "Cray",
    "Curran", "Darwin", "Davinci", "Dewdney", "Dhawan", "Diffie",
    "Driscoll", "Dubinsky", "Easley", "Edison", "Einstein", "Elbakyan",
    "Elgamal", "Elion", "Ellis", "Engelbart", "Euler", "Feistel",
    "Fermat", "Fermi", "Gagarin", "Ganguly", "Gates", "Gauss",
    "Germain", "Goldberg", "Goldstine", "Goldwasser", "Golick", "Goodall",
    "Gould", "Greider", "Grothendieck", "Haibt", "Hamilton", "Haslett",
    "Hawking", "Hellman", "Heisenberg", "Hermann", "Herschel", "Hertz",
    "Heyrovsky", "Hodgkin", "Hofstadter", "Hoover", "Hugle", "Hypatia",
    "Ishizaka", "Jackson", "Jang", "Jennings", "Jepsen", "Johnson",
    "Joliot", "Jones", "Kalam", "Kapitsa", "Kare", "Keldysh",
    "Keller", "Khayyam", "Khorana", "Kilby", "Kirch", "Kowalevski",
    "Lalande", "Leakey", "Leavitt", "Lederberg", "Lehmann", "Lewin",
    "Lichterman", "Liskov", "Lumiere", "Mahavira", "Margulis", "Matsumoto",
    "Mayer", "McLaren", "McLean", "McNulty", "Mendel", "Mendeleev",
    "Meninsky", "Merkle", "Mestorf", "Mirzakhani", "Moore", "Morse",
    "Murdock", "Moser", "Napier", "Nash", "Newton", "Nightingale",
    "Nobel", "Northcutt", "Noyce", "Panini", "Pare", "Pasteur",
    "Payne", "Perlman", "Pike", "Poitras", "Proskuriakova", "Ptolemy",
    "Ramanujan", "Ride", "Montalcini", "Rhodes", "Robinson", "Roentgen",
    "Rosalind", "Rubin", "Saha", "Sammet", "Sanderson", "Satoshi",
    "Shamir", "Shaw", "Shirley", "Shockley", "Shtern", "Sinoussi",
    "Snyder", "Solomon", "Spence", "Stonebraker", "Sutherland", "Swanson",
    "Swartz", "Swirles", "Taussig", "Tereshkova", "Tesla", "Tharp",
    "Thompson", "Torvalds", "Tu", "Varahamihira", "Visvesvaraya", "Volhard",
    "Villani", "Wescoff", "Wilbur", "Wiles", "Williams", "Williamson",
    "Wilson", "Wing", "Wozniak", "Wright", "Wu", "Yalow",
    "Yonath", "Zhukovsky",

    # Additional astronomical codenames
    "Achernar", "Acrux", "Adhara", "Aegir", "Aether", "Agena", "Algenib",
    "Algol", "Alioth", "Alkaid", "Alpheratz", "Ankaa", "Ara", "Atlas",
    "Atria", "Beid", "Chertan", "Corvus", "Delphinus", "Diadem", "Diphda",
    "Dubhe", "Elnath", "Fomalhaut", "Hadar", "Hamal", "Izar", "Kaus",
    "Kochab", "Markab", "Matar", "Merak", "Merope", "Mira", "Mirach",
    "Mizar", "Musca", "Navi", "Nunki", "Ophiuchus", "Pherkad", "Rasalas",
    "Rastaban", "Sadr", "Sargas", "Scorpius", "Shaula", "Sheratan", "Skat",
    "Unukalhai", "Vindemiatrix", "Wezen", "Zaurak", "Zubenelgenubi",
    "Adrastea", "Aitne", "Albiorix", "Bebhionn", "Bergelmir", "Bestla",
    "Calypso", "Carme", "Daphnis", "Fenrir", "Fornjot", "Greip", "Hati",
    "Hegemone", "Helene", "Hermippe", "Ijiraq", "Janus", "Kari", "Kiviuq",
    "Loge", "Mundilfari", "Narvi", "Paaliaq", "Pandora", "Polydeuces",
    "Prometheus", "Siarnaq", "Surtur", "Sycorax", "Tarazed", "Alpherg",
    "Caph", "Menkent", "Ruchbah",

    # Easter eggs
    "Stepjam", "Favour", "Nwachukwu", "Onion", "Aditya", "Wagh", "Couagroo", "Tasker", 
    "Damon", "Hayhurst", "Dennis", "Thevara", "Felix", "Bartlett", "Mark", 
    "Naeem", "Muneeb", "Amer", "Sarthak", "Das", "Stephen", "James", "Steven", 
    "Jacobs", "Yiklung", "Pang", "BeepBoop", "RobotWhisperer", "ThreeLaws", 
    "Rufus", "McDonald", "CrystalBall", "Sandros"
]

# cspell:enable
# fmt: on


def generate_recording_name() -> str:
    """Generate a memorable two-word recording codename."""
    return (
        f"{random.choice(RECORDING_DESCRIPTORS)} "
        f"{random.choice(RECORDING_CODENAMES)}"
    )
