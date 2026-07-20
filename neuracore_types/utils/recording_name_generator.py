"""Generate memorable, readable two-word recording codenames.

Each generated name combines a standalone technical descriptor with a familiar
codename. The vocabulary deliberately avoids joined CamelCase terms so names
remain easy to read aloud and remember.
"""

import random

# ruff: noqa: E501
# fmt: off
# cspell:disable

RECORDING_DESCRIPTORS = [
    # Mathematics and structure
    "Adaptive", "Algebraic", "Analytic", "Angular", "Arched",
    "Arithmetic", "Apex", "Basic", "Banded", "Bilinear",
    "Binary", "Bounded", "Canonical", "Cardinal", "Cartesian", "Causal",
    "Chaotic", "Circular", "Combined", "Common", "Complex",
    "Composite", "Continuous", "Convex", "Paired", "Cubic", "Curved",
    "Cyclic", "Decimal", "Differential", "Dimensional", "Discrete",
    "Dynamic", "Elliptic", "Even", "Exact",
    "Exponential", "Factor", "Finite", "Fractional", "Fractal",
    "Functional", "Geometric", "Gradient", "Harmonic",
    "Helical", "Curving", "Integral", "Invariant",
    "Equal", "Iterative", "Linear", "Logical",
    "Linked", "Matrix", "Metric", "Modular", "Natural",
    "Nonlinear", "Numerical", "Orbital", "Parallel",
    "Tunable", "Periodic", "Planar", "Polar", "Polynomial", "Prime",
    "Probable", "Quadratic", "Radial", "Rational", "Recursive",
    "Relational", "Rotational", "Scalar", "Smooth", "Spectral",
    "Statistical", "Steady", "Symmetric", "Tensor", "Mapped",
    "Mutual", "Single", "Varied", "Vector", "Pointed",
    "Rippled",

    # Robotics and control
    "Agile", "Aligned", "Articulated", "Autonomous", "Balanced",
    "Calibrated", "Capable", "Closed", "Compliant", "Controlled",
    "Coordinated", "Decisive", "Deterministic", "Dexterous",
    "Direct", "Distributed", "Embodied", "Emergent", "Feedback", "Flexible",
    "Focused", "Guided", "Haptic", "Integrated",
    "Interactive", "Kinetic", "Steered", "Measured",
    "Machine", "Mobile", "Modulated", "Blended", "Navigable",
    "Nimble", "Observable", "Optimized", "Perceptive", "Precise",
    "Predictive", "Reactive", "Redundant", "Regulated", "Reliable",
    "Resilient", "Responsive", "Robotic", "Robust", "Servo", "Stabilized",
    "Synchronized", "Tactile", "Routed", "Temporal", "Tuned",
    "Versatile", "Vigilant", "Weighted",

    # Engineering and physics
    "Acoustic", "Aerodynamic", "Analog", "Atomic", "Ceramic",
    "Chromatic", "Coherent", "Conductive", "Copper", "Digital", "Electrical",
    "Electronic", "Elastic", "Charged", "Embedded", "Energetic", "Fluid",
    "Hydraulic", "Magnetic", "Mechanical", "Metallic", "Optical",
    "Quartz", "Radiant", "Resonant", "Rigid", "Sonic",
    "Thermal", "Titanium", "Turbulent", "Ultrasonic", "Vibrant", "Viscous",
    "Wireless", "Ballistic", "Capillary", "Cohesive", "Coupled",
    "Crystalline", "Deformable", "Diffusive", "Electric",
    "Frictional", "Inductive", "Lubricated", "Material",
    "Vibrating", "Plastic", "Structured", "Pressurized", "Propulsive", "Pulsed",
    "Resistive", "Evolved", "Sealed", "Solid", "Structural",

    # Computing and systems
    "Ordered", "Buffered", "Cached", "Clear", "Compiled", "Built",
    "Joint", "Connected", "Consistent", "Defined", "Encoded", "Encrypted",
    "Joined", "Fused", "Granular", "Hybrid", "Fixed", "Indexed",
    "Inferred", "Readable", "Isolated", "Lean", "Local", "Memoryless",
    "Networked", "Open", "Flowing", "Portable", "Programmable", "Stepped",
    "Queued", "Remote", "Composed", "Reusable", "Safe", "Scalable", "Secure",
    "Sequential", "Shared", "Simple", "Sparse", "Grounded", "Untied",
    "Streaming", "Symbolic", "Timed", "Threaded", "Coded", "Trusted",
    "Typed", "Unified", "Validated", "Virtual",

    # Space, materials, and geometry
    "Adjacent", "Aerial", "Amber", "Astral", "Axial", "Azure", "Celestial",
    "Cobalt", "Compact", "Concentric", "Cosmic", "Crimson", "Arced",
    "Diagonal", "Directional", "Spaced", "Galactic", "Global", "Golden",
    "Horizontal", "Indigo", "Starry", "Glowing", "Jade", "Lateral",
    "Layered", "Luminous", "Lunar", "Martian", "Mirrored", "Nebular", "Oblique",
    "Obsidian", "Opaque", "Prismatic", "Proximal", "Reflective", "Refractive",
    "Sapphire", "Silver", "Solar", "Spatial", "Spherical", "Stellar",
    "Transparent", "Upright", "Vertical", "Violet", "Central", "Crystal",
    "Equatorial", "Exterior", "Graphite", "Interior", "Ivory", "Contour",
    "Meridian", "Monochrome", "Nocturnal", "Northern", "Peripheral", "Platinum",
    "Southern", "Blue", "Visible",

    # Reliability and readable technical language
    "Active", "Advanced", "Alert", "Bright", "Bold", "Brisk", "Calm", "Careful",
    "Clean", "Crisp", "Dependable", "Diligent", "Durable", "Economical",
    "Efficient", "Elegant", "Fast", "Graceful", "Intuitive", "Lucid",
    "Methodical", "Orderly", "Patient", "Polished", "Practical", "Ready",
    "Refined", "Repeatable", "Renewed", "Sharp", "Stable", "Strong", "Swift",
    "Systematic", "Timely", "Trusty", "Useful", "Abundant", "Applied", "Branching",
    "Tiered", "Clustered", "Collective", "Shaped", "Conservative", "Convergent",
    "Dense", "Derived", "Divergent", "Damped", "Eventful", "Explicit", "Feasible",
    "Focal", "General", "Generative", "Gentle", "Incremental", "Independent",
    "Indirect", "Informed", "Instant", "Intelligent", "Latent", "Minimal",
    "Molecular", "Neutral", "Nominal", "Normal", "Novel", "Nuclear", "Operating",
    "Organic", "Passive", "Positive", "Potential", "Primary", "Pure", "Quiet", "Rapid",
    "Regular", "Relative", "Reversible", "Rounded", "Rich", "Selective", "Semantic",
    "Serial", "Silent", "Standard", "Static", "Straight", "Subtle", "Wide",
    "Broad", "Undamped", "Uniform", "Variable", "Whole",
]

RECORDING_CODENAMES = [
    # Space and astronomy
    "Aero", "Aster", "Astra", "Atlas", "Aurora", "Apollo", "Alder",
    "Aries", "Arlo", "Blaine", "Brant", "Ariel", "Argo", "Astro", "Avalon",
    "Brice", "Carrick", "Cates",
    "Comet", "Cosmos", "Clive", "Draco", "Eclipse", "Electra",
    "Equinox", "Europa", "Galaxy", "Cobb", "Halley", "Helios", "Horizon",
    "Cormac", "Jupiter", "Luna", "Darby", "Mars", "Mercury", "Meteor", "Nebula",
    "Neptune", "Nova", "Dax", "Orion", "Pegasus", "Polaris", "Doran",
    "Pulsar", "Quasar", "Dorsey", "Saturn", "Sol", "Solaris",
    "Stellar", "Titan", "Elwood", "Umbra", "Uranus", "Vega", "Vela", "Venus",
    "Vulcan", "Zenith", "Faber",

    # Mythology and classic codenames
    "Achilles", "Adonis", "Ajax", "Ares", "Artemis", "Athena", "Fallon",
    "Calypso", "Firth", "Cupid", "Gaines", "Diana", "Echo",
    "Freya", "Gaia", "Hector", "Hera", "Hermes", "Iris", "Juno",
    "Lancelot", "Loki", "Merlin", "Midas", "Minerva", "Gable", "Gentry",
    "Nike", "Grady", "Odin", "Olympus", "Harlan", "Pandora", "Hayes",
    "Poseidon", "Prometheus", "Huxley", "Robin", "Roland", "Rune", "Thor", "Ives",
    "Vesta",

    # Science, mathematics, computing, and robotics pioneers
    "Abel", "Ada", "Allen", "Almeida", "Antonelli", "Archimedes",
    "Aristotle", "Asimov", "Armstrong", "Austin", "Babbage", "Bardeen",
    "Bartik", "Bassi", "Bayes", "Beaver", "Bell", "Benz",
    "Blackwell", "Bose", "Boyd", "Raines", "Brown",
    "Buck", "Burnell", "Cannon", "Carson", "Carver", "Cerf", "Jory",
    "Kade", "Church", "Clarke", "Cohen", "Copernicus", "Cori", "Cray",
    "Curie", "Curran", "Darwin", "Devol", "Kellan",
    "Dubinsky", "Easley", "Edison", "Einstein", "Elion", "Ellis",
    "Reeve", "Faraday", "Kirk", "Fermi", "Fermat",
    "Franklin", "Gagarin", "Galileo", "Gates", "Germain",
    "Goldberg", "Goodall", "Gould", "Greider", "Hamilton", "Hawking", "Hellman", "Hertz",
    "Herschel", "Hilbert", "Hodgkin", "Hopper", "Hubble", "Landon", "Jackson",
    "Jemison", "Jennings", "Johnson", "Joliot", "Jones", "Joule", "Kalam",
    "Kare", "Kepler", "Keller", "Lyle", "Kilby", "Lamarr", "Lamport", "Lalande",
    "Leakey", "Leavitt", "Lederberg", "Lehmann", "Lewin", "Liskov", "Lovelace",
    "Mahavira", "Marconi", "Margulis", "Matsumoto", "Maxwell", "Mayer",
    "McCarthy", "McClintock", "McLaren", "Meitner", "Mendel", "Rook", "Merkle",
    "Madden", "Minsky", "Moore", "Moravec", "Morse", "Moser", "Napier", "Nash", "Newton",
    "Nightingale", "Nilsson", "Noyce", "Pascal", "Pasteur", "Payne", "Perlman",
    "Pike", "Planck", "Mace", "Raman", "Niles", "Ride", "Ritchie",
    "Robinson", "Rosenblatt", "Rubin", "Sagan", "Saha", "Sammet", "Sanderson", "Satoshi",
    "Shannon", "Shamir", "Shaw", "Shirley", "Shockley", "Simon", "Snyder", "Solomon",
    "Spence", "Nolan", "Oak", "Swanson", "Swartz", "Tesla", "Tharp", "Thompson",
    "Pace", "Turing", "Vaughan", "Villani", "Volta", "Watt", "Wiener", "Wiles",
    "Williams", "Williamson", "Wilson", "Wing", "Phelps", "Wright", "Wu", "Yalow", "Price",
    "Zuse",

    # Familiar, easy-to-say surnames
    "Ainsley", "Albright", "Alcott", "Aldridge", "Atkinson", "Baldwin", "Bancroft",
    "Barrow", "Beale", "Bedford", "Bentley", "Bingham", "Bolton", "Bradford", "Bramwell",
    "Brandon", "Brewer", "Bromley", "Buckley", "Callahan", "Carleton", "Carrington",
    "Clay", "Clements", "Colton", "Corbett", "Cross", "Cummings", "Dale", "Darrow",
    "Denham", "Ellwood", "Fairbanks", "Faulkner", "Fenwick", "Fitzroy", "Fleming", "Forde",
    "Gifford", "Goodwin", "Graham", "Hadfield", "Hale", "Hall", "Halstead", "Hanley",
    "Hargrove", "Hayward", "Heath", "Hensley", "Hinton", "Hodges", "Hollis", "Howarth",
    "Hudson", "Keane", "Kenyon", "Kirby", "Langford", "Larkin", "Latham", "Lawton",
    "Linden", "Lockwood", "Marlowe", "Maynard", "Melton", "Merton", "Moffat", "Montrose",
    "Neville", "Norwood", "Oakley", "Osborne", "Page", "Parnell", "Pearson", "Pendleton",
    "Pollard", "Prior", "Randall", "Rawlings", "Redmond", "Roper", "Royce", "Sadler",
    "Sandford", "Sargent", "Sheridan", "Slater", "Sloane", "Somerville", "Stafford",
    "Stanford", "Stroud", "Talbot", "Templeton", "Thornton", "Townsend", "Vickers",
    "Vincent", "Walton", "Warner", "Waverly", "Wayland", "Whitman", "Wilcox", "Winters",
    "Withers", "Woodson", "Worth", "Yarrow",

    # Familiar places, surnames, and neutral codenames
    "Aegis", "Alpha", "Anchor", "Arc", "Arrow", "Bolt", "Bridge", "Cairn", "Crown", "Dart",
    "Drift", "Edge", "Fable", "Flare", "Forge", "Frontier", "Gate", "Glide", "Halo", "Helix",
    "Hollow", "Jolt", "Key", "Kite", "Lumen", "Mosaic", "North", "Oasis", "Pillar", "Pilot",
    "Quest", "Rally", "Relay", "Sail", "Spark", "Spire", "Torch", "Trail", "Vault", "Vista",
    "Voyager", "Abbey", "Acadia", "Arbour", "Arden", "Ashby", "Brighton", "Bristol", "Camden",
    "Carlisle", "Caspian", "Chelsea", "Chester", "Clifton", "Conway", "Dover", "Durham", "Eden",
    "Eldon", "Essex", "Fairmont", "Grafton", "Hadley", "Harrow", "Hartley", "Henley", "Keswick",
    "Kendal", "Kingsley", "Langley", "Lincoln", "Mariner", "Milton", "Oxford", "Penrose", "Preston",
    "Radcliffe", "Raleigh", "Ramsay", "Ripley", "Rowan", "Sheldon", "Somers", "Sutton", "Telford",
    "Trent", "Waverley", "Weston", "Whitby", "Windsor", "York", "Alden", "Alfred", "Alma", "Arlen",
    "Arthur", "Avery", "Bailey", "Blair", "Blake", "Brady", "Cameron", "Carter", "Casey", "Cole",
    "Connor", "Cooper", "Dana", "Devon", "Drew", "Evan", "Finn", "Flynn", "Fraser", "Grant", "Grey",
    "Harley", "Harper", "Hayden", "Jamie", "Jordan", "Kendall", "Lane", "Logan", "Mason", "Morgan",
    "Parker", "Quinn", "Reese", "Riley", "Sawyer", "Scott", "Taylor", "Tyler", "Walker", "Warren",
    "Westley", "Ames", "Archer", "Barker", "Barton", "Baxter", "Benson", "Bishop", "Blaise",
    "Caldwell", "Campbell", "Canton", "Cavendish", "Cromwell", "Dalton", "Darcy", "Dawson", "Denton",
    "Easton", "Ellington", "Emerson", "Fairchild", "Farrell", "Fletcher", "Forbes", "Ford", "Garner",
    "Gilbert", "Gordon", "Griffin", "Harding", "Harrison", "Hawthorne", "Henderson", "Hughes", "Hunter",
    "Ingram", "Irwin", "Jarvis", "Jensen", "Kendrick", "King", "Lawson", "Leonard", "Lennox", "Manning",
    "Marsh", "Merritt", "Noble", "Norris", "Palmer", "Perry", "Porter", "Raymond", "Reed", "Reid",
    "Ross", "Russell", "Sinclair", "Spencer", "Sterling", "Tanner", "Turner", "Underwood", "Vernon",
    "Watson", "Webster", "Wheeler", "Wilder", "Woodward", "Yates",

    # Easter eggs
    "Stepjam", "Favour", "Nwachukwu", "Onion", "Aditya", "Wagh", "Couagroo", "Tasker", 
    "Damon", "Hayhurst", "Dennis", "Thevara", "Felix", "Bartlett", "Mark", 
    "Naeem", "Muneeb", "Amer", "Sarthak", "Das", "Stephen", "James", "Steven", 
    "Jacobs", "Yiklung", "Pang","Rufus", "McDonald", "Sandros"
]

# cspell:enable
# fmt: on


def generate_recording_name() -> str:
    """Generate a memorable two-word recording codename."""
    return (
        f"{random.choice(RECORDING_DESCRIPTORS)} "
        f"{random.choice(RECORDING_CODENAMES)}"
    )
