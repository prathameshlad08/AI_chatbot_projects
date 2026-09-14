import xml.etree.ElementTree as ET
from pathlib import Path
import json
OUTPUT_FILE = Path("data/medquad_parsed.json")


def parse_file(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    focus_elem = root.find("Focus")
    focus = focus_elem.text.strip() if focus_elem is not None and focus_elem.text else "Unknown"

    qa_pairs = []
    for qapair in root.findall(".//QAPair"):
        question_elem = qapair.find("Question")
        answer_elem = qapair.find("Answer")

        question_text = question_elem.text if question_elem is not None else None
        qtype = question_elem.get("qtype") if question_elem is not None else None
        answer_text = answer_elem.text if answer_elem is not None else None

        if not question_text or not answer_text or not answer_text.strip():
            continue

        qa_pairs.append({
            "focus": focus,
            "qtype": qtype,
            "question": question_text.strip(),
            "answer": answer_text.strip()
        })

    return qa_pairs


if __name__ == "__main__":
    sample = parse_file("MedQuAD/1_CancerGov_QA/0000001_1.xml")
    for qa in sample:
        print(qa)
        print("---")




def parse_all(medquad_dir):
    all_qa_pairs = []
    xml_files = list(Path(medquad_dir).rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files")

    for i, filepath in enumerate(xml_files):
        try:
            qa_pairs = parse_file(filepath)
            all_qa_pairs.extend(qa_pairs)
        except (ET.ParseError, AttributeError) as e:
            print(f"Skipping malformed file {filepath}: {e}")

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(xml_files)} files...")

    return all_qa_pairs



if __name__ == "__main__":
    all_qa = parse_all("MedQuAD")
    print(f"\nTotal QA pairs extracted: {len(all_qa)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2)

    print(f"Saved to {OUTPUT_FILE}")
