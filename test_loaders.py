from preprocess_loaders import load_pdf, load_text, ocr_image, split_documents

print("Testing PDF loader...")
pdf_docs = load_pdf("data/pdfs/example_pdf.pdf")
print(pdf_docs[0].page_content[:300], "\n")

print("Testing text loader...")
txt_docs = load_text("data/text/example_txt.txt")
print(txt_docs[0].page_content[:300], "\n")

print("Testing image OCR loader...")
img_docs = ocr_image("data/images/example_img.jpg")
print(img_docs[0].page_content[:300], "\n")

print("Testing chunking...")
chunks = split_documents(pdf_docs)
print("Chunks:", len(chunks))
print("First chunk preview:", chunks[0].page_content[:200])
