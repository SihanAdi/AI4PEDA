import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import os
import pdfplumber
from tqdm import tqdm


def detect_header_footer(pdf_document, threshold=200):
    header_footer = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_array = np.array(img.convert('L'))

        header_height = next((y for y in range(img_array.shape[0]) if np.mean(img_array[y, :]) < threshold), 0)
        footer_height = next((y for y in range(img_array.shape[0] - 1, -1, -1) if np.mean(img_array[y, :]) < threshold),
                             0)
        footer_height = img_array.shape[0] - footer_height if footer_height else 0

        header_footer.append((header_height, footer_height))
    return header_footer


def extract_block_diagram_section(pdf_document, output_image_path, header_footer, keyword="Block Diagram"):
    img_index = 0
    for page_num in range(len(pdf_document)):
        occurrences = []
        page = pdf_document.load_page(page_num)
        header_height, footer_height = header_footer[page_num]
        page_height = page.rect.height

        text_instances = page.search_for(keyword)
        for inst in text_instances:
            if header_height <= inst.y1 <= (page_height - footer_height):
                occurrences.append((page_num, inst.y1))

        if not occurrences:
            continue

        cropped_images = []
        for i in range(len(occurrences) - 1):
            page_num_1, y1 = occurrences[i]
            page_num_2, y2 = occurrences[i + 1]

            if page_num_1 == page_num_2 and (y2 - y1) > 200:
                page = pdf_document.load_page(page_num_1)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_array = np.array(img)
                cropped_img_array = img_array[int(y1):int(y2), :]
                cropped_images.append(Image.fromarray(cropped_img_array))

        if cropped_images:
            for cropped_img in cropped_images:
                os.makedirs(output_image_path, exist_ok=True)
                cropped_img.save(f"{output_image_path}/part_{img_index}.png")
                print(f"Page {page_num} Block Diagram section saved to {output_image_path}/part_{img_index}.png")
                img_index += 1
            return True
    return False


def extract_text_without_headers_footers(pdf_path, keyword, header_footer, threshold=200):
    extracted_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_height = page.height
            text = page.extract_text(x_tolerance=1)
            header_height, footer_height = header_footer[page_num]
            if keyword in text:
                print(f"Keyword '{keyword}' found on page {page_num + 1}")
                for block in page.extract_words(x_tolerance=1):
                    block_top = block['top']
                    block_bottom = block['bottom']
                    if block_top > header_height and block_bottom < (page_height - footer_height):
                        extracted_text += block['text'] + " "
    return extracted_text


def process_pdfs_in_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.pdf'):
            print(f"\nProcessing {filename}")
            pdf_path = os.path.join(folder_path, filename)
            pdf_name = os.path.splitext(filename)[0]
            pdf_output_folder = os.path.join(output_folder, pdf_name)

            if not os.path.exists(pdf_output_folder):
                os.makedirs(pdf_output_folder)

            pdf_document = fitz.open(pdf_path)
            header_footer = detect_header_footer(pdf_document)

            system_description = extract_text_without_headers_footers(pdf_path, "System Description", header_footer)
            if system_description:
                text_file_path = os.path.join(pdf_output_folder, "System_Description.txt")
                with open(text_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(system_description)
                print(f"System Description saved to {text_file_path}")
            else:
                print(f"System Description not found in {filename}")

            if not extract_block_diagram_section(pdf_document, pdf_output_folder, header_footer,
                                                 keyword="Block Diagram"):
                extract_block_diagram_section(pdf_document, pdf_output_folder, header_footer, keyword="方框图")


if __name__ == "__main__":
    pdf_path = "input/"
    output_path = "output/"
    process_pdfs_in_folder(pdf_path, output_path)
