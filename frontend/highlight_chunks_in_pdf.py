import fitz  # PyMuPDF
import subprocess
from rapidfuzz import fuzz, process  # faster and better than fuzzywuzzy


def open_page_chunk(pdf_path,chunk):
    doc = fitz.open(pdf_path)
    acrobat_path = r"C:\Program Files\Adobe\Acrobat DC\Acrobat\Acrobat.exe"
    sumatra_path = r"D:\SumatraPDF\SumatraPDF.exe"
    page_num = chunk.metadata.get("page")
    ''' subprocess.Popen([
        acrobat_path,
        '/A',
        f'page={page_num+1}=OpenActions',
        pdf_path
    ]) '''

    subprocess.Popen([
        sumatra_path,
        f'-page', str(page_num + 1),
        pdf_path
    ])


def open_page_chunk_annot(pdf_path, chunk, user_query, output_pdf_path="annotated_output.pdf", threshold=35):
    """
    Uses a sliding 4-word window on the chunk text, compares to query using fuzzy match,
    and highlights the actual matching text found on the PDF page.
    """
    doc = fitz.open(pdf_path)
    page_num = chunk.metadata.get("page")
    page = doc[page_num]

    # Raw chunk text from the page, cleaned
    raw_text = chunk.page_content.replace('\n', ' ').strip()
    words = raw_text.split()
    window_size = 4

    matches_found = 0

    for i in range(len(words) - window_size + 1):
        window_text = ' '.join(words[i:i + window_size])
        score = fuzz.partial_ratio(window_text.lower(), user_query.lower())

        if score >= threshold:
            # Try to find that exact text on the actual PDF page
            rects = page.search_for(window_text)
            if rects:
                print(f"✅ Match (score {score}): \"{window_text}\"")
                matches_found += 1
                for rect in rects:
                    annot = page.add_highlight_annot(rect)
                    annot.set_info(content=f"Matched: {user_query}")

    # Save the annotated file
    doc.save(output_pdf_path)
    doc.close()

    # Open the annotated file in SumatraPDF at correct page
    sumatra_path = r"D:\SumatraPDF\SumatraPDF.exe"
    subprocess.Popen([
        sumatra_path,
        '-page', str(page_num + 1),
        output_pdf_path
    ])

    if matches_found == 0:
        print("No matching text found on the actual PDF page.")


def highlight_chunks_in_pdf(pdf_path, chunks, output_path="highlighted_rulebook.pdf"):
    doc = fitz.open(pdf_path)
    for chunk in chunks:
        try:
            page_num = chunk.metadata.get("page")
            text = chunk.page_content.strip()

            if not page_num or not text:
                print(f"⚠️ Skipping chunk with no page/text")
                continue

            page_index = int(page_num) - 1
            page = doc[page_index]
            page_text = page.get_text()
            clean_chunk = ' '.join(text.split())

            print(f"\n🔍 Searching on page {page_num}...")

            # --- Try full exact match ---
            matches = page.search_for(clean_chunk)

            # --- Fallback 1: first 100 chars ---
            if not matches:
                short_text = clean_chunk[:100]
                matches = page.search_for(short_text)
                if matches:
                    print(f"✅ Found partial match (first 100 chars) on page {page_num}")

            # --- Fallback 2: fuzzy matching line-by-line ---
            if not matches:
                lines = page_text.splitlines()
                fuzzy_match = difflib.get_close_matches(short_text, lines, n=1, cutoff=0.7)
                if fuzzy_match:
                    print(f"✅ Found fuzzy match on page {page_num}: '{fuzzy_match[0]}'")
                    matches = page.search_for(fuzzy_match[0])

            # --- Highlight all matches ---
            if matches:
                print(f"✨ Highlighting {len(matches)} match(es) on page {page_num}")
                for rect in matches:
                    page.add_highlight_annot(rect)
            else:
                print(f"❌ No match found for chunk on page {page_num}")

        except Exception as e:
            print(f"❌ Error on page {chunk.metadata.get('page')}: {e}")

    doc.save(output_path)
    print(f"\n📄 Highlighted PDF saved as: {output_path}")
