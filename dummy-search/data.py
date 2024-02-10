from classes import Dataset

basketball = Dataset(
    name="Basketball",
    color="#26828e",
    keyword_allow_list=["Basketball"],
    allow_list=[
        "1991107619818",
        "19753803125",
        "19753803009",
        "199110766513",
        "1991107619129",
        "1991107616835",
        "19753803184",
        "1991107619479",
        "19753803183",
        "1991107619410",
        "199110767086",
        "1991107612031",
        "1991107619686",
        "1991107619798",
        "1991107619799",
        "19753803046",
        "19753803057",
        "19753803081",
        "19753803041",
        "1991107610782",
        "199110767085",
        "19753803070",
    ],
)
baseball = Dataset(
    name="Baseball",
    color="#a0da39",
    keyword_allow_list=[
        "baseball",
        "ball team",
        "ball players",
        "cardinals",
        "ball tournament",
        "white sox",
        "lethbridge north stars",
        "ball park",
        "ball field",
        "fastball team",
        "hardball team",
        "Little League",
    ],
    allow_list=[
        "199110769122",
        "199110768824",
        "199110767797",
        "1991107617706",
        "19753802006",
        "199110769125",
        "1991107618048",
        "19753802084",
        "199110767502",
        "19753802135",
        "1991107617705",
        "19753802081",
        "1991107618205",
        "19753802090",
        "19753802082",
        "19753802097",
        "19753802101",
        "19752311054",
    ],
)
football = Dataset(
    name="Football",
    color="#4ac16d",
    keyword_allow_list=[
        "quarterback",
        "football",
        "touchdown",
        "interception",
        "sack",
        "fumble",
        "punt",
        "hail mary",
        "snap",
        "end zone",
        "field goal",
    ],
    allow_list=[],
    block_list=[],
)
soccer = Dataset(
    name="Soccer",
    color="#1fa187",
    keyword_allow_list=["soccer", "European football"],
)
women = Dataset(
    name="Women",
    color="#277f8e",
    keyword_allow_list=["woman", "women", "female", "wife", "queen", "she"],
)
men = Dataset(
    name="Men",
    color="#365c8d",
    keyword_allow_list=["man", "men", "male", "husband", "king", "he"],
)
cars = Dataset(
    name="Auto",
    color="#46327e",
    keyword_allow_list=[
        "auto",
        "car",
        "cars",
        "vehicle",
        "vehicles",
        "automobile",
        "automobiles",
        "parking lot",
        "tire",
        "truck",
        "train-truck",
        "tractor",
        # Car brands
        "Ford",
        "Chrysler",
        "Duesenberg",
        "Chevrolet",
        "Chevy",
        "Willys-Overland",
        "Tucker",
        "Oldsmobile",
        "Cadillac",
        "Mustang",
        "Camaro",
        "Pontiac",
        "AMC",
        "Dodge",
        "Buick",
        "Alfa Romeo",
        "Jaguar",
        "British Leyland",
        "Ferrari",
        "Porsche",
        "BMW",
        "Plymouth",
        "Mercedes-Benz",
        "Lotus",
        "Datsun",
        "Lamborghini",
        "Volvo",
        "Studebaker",
        "Jeep",
        "Road Patrol",
        "motorist",
        "motorists",
        "1956 model",
    ],
    allow_list=[
        "1991107618405",
        "19752608087b",
        "199110767246",
        "199110768726",
        "199110769073",
        "1991107617414",
        "199110761890",
        "1991107619903",
        "199110768722",
        "19752303298",
        "19752303223",
        "199110765469",
        "199110769074",
        "19752606016",
        "199110767324",
        "199110769072",
        "199110766700",
        "199110769713",
        "19753500006",
        "19752390781",
        "19752901030",
    ],
    block_list=[
        "train",
    ]
)
indian = Dataset(
    name="First Nations",
    color="#fde725",
    keyword_allow_list=[
        "indian",
        "native",
        "first nations",
        # "chief",
        # "ceremony",
        "Indigenous",
        "tribe",
        "Kainai",
        "blood tribe",
        "traditional dress",
        "Blood Reserve",
        "headdress",
        "Red Bull Society",
    ],
    keyword_block_list=["Lethbridge Native Sons"],
    allow_list=[
        "1991107618702",
        "1991107618703",
        "19754011095",
    ],
    block_list=[
        "199110765217",
        "19752903015",
        "19752904007",
        "19752908144",
        "19752908145",
        "19753000008",
        "19753000082",
        "19753000081",
    ],
)

datasets: list[Dataset] = [
    basketball,
    baseball,
    football,
    soccer,
    cars,
    indian,
    women,
    men,
]
