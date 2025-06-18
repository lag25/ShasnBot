import fitz  # PyMuPDF
import difflib

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
