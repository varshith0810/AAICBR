from dataclasses import dataclass
from pathlib import Path

BREEDS = [
    "vechur","umblachery","toda","tharparkar","surti","sahiwal","redsindhi","reddane",
    "rathi","pulikulam","ongole","nimari","niliravi","nagpuri","nagori","murrah",
    "mehsana","malnadgidda","krishnavalley","khillari","kherigarh","kenkatha","kasargod",
    "kankrej","kangayam","jersey","jaffrabadi","holsteinfriesian","hariana","hallikar",
    "guernsey","gir","deoni","dangi","brownswiss","bhadawari","bargur","banni",
    "ayrshire","amritmahal","alambadi"
]


@dataclass
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_zip: Path = project_root / "dataset.zip"
    extracted_data: Path = project_root / "data" / "breeds"
    model_dir: Path = project_root / "models"
    output_dir: Path = project_root / "outputs"
