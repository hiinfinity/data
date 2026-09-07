from pathlib import Path
from booknlp.booknlp import BookNLP

model_params = {
    "pipeline": "entity,event",
    "model": "small"
}

booknlp = BookNLP("en", model_params)

input_dir = Path("/media/secure_volume/books")
output_root = Path("/media/secure_volume/booknlp_output")
output_root.mkdir(parents=True, exist_ok=True)

for txt_file in input_dir.rglob("*.txt"):
    book_id = txt_file.stem
    output_dir = output_root / book_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {txt_file}")

    try:
        booknlp.process(
            str(txt_file),
            str(output_dir),
            book_id
        )
        print(f"Completed: {book_id}")
    except Exception as e:
        print(f"ERROR {book_id}: {e}")
