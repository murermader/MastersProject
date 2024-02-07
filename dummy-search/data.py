from classes import Dataset

basketball = Dataset(
    name="Basketball",
    keywords=["Basketball"],
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
    block_list=[],
)

baseball = Dataset(
    name="Baseball",
    keywords=[
        "baseball",
        "ball team",
        "ball players",
        "cardinals",
        "ball tournament",
        "white sox",
        "lethbridge north stars",
        "ball park",
        "ball field",
        "fastball team" "hardball team",
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
    block_list=[],
)

football = Dataset(
    name="Football",
    keywords=[
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
    keywords=["soccer", "European football"],
    allow_list=[],
    block_list=[],
)

# basketball_extended = Dataset(
#     "Basketball Extended",
#     basketball.keywords,
#     basketball.allow_list.union(
#         set(
#             "1991107616835",  # Photo of an emtpy gymnasium (visible basketball hoop and ground markings)
#         )
#     ),
#     [],
# )

indian = Dataset(
    name="Indian",
    keywords=[
        "indian",
        "native",
        "first nations",
        "chief",
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
    allow_list=[
        "1991107618702",
        "1991107618703",
        "19754011095",
    ],
    block_list=[],
)

datasets: list[Dataset] = [
    basketball,
    baseball,
    football,
    soccer,
    indian,
]
