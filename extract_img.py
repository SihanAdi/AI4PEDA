import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import os
import pdfplumber
from tqdm import tqdm
from openai import OpenAI
import base64
import configparser


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


def run_model(model, img_question, base64_img):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'user', 'content': [
                {'type': 'text', 'text': img_question},
                {'type': 'image_url', 'image_url': {
                    'url': f"data:image/PNG;base64,{base64_img}"
                }}
            ]}
        ]
    )
    return response.choices[0].message.content


def encode_img(path):
    with open(path, 'rb') as img:
        return base64.b64encode(img.read()).decode('utf-8')


def process_png(file_path):
    with Image.open(file_path) as img:
        base64_img = encode_img(file_path)
        return run_model(MODEL, '请提取图片中的主要模块，并列出图里所有连线的连接关系', base64_img)


def save_text_to_file(text, file_path):
    with open(file_path, 'w') as file:
        file.write(text)


def traverse_and_process_png(root_dir):
    missing_png_folders = []
    missing_txt_folders = []
    empty_folders = []

    for item in os.listdir(root_dir):
        subdir = os.path.join(root_dir, item)
        if os.path.isdir(subdir):
            png_files = [file for file in os.listdir(subdir) if file.endswith('.png')]
            txt_files = [file for file in os.listdir(subdir) if file.endswith('.txt')]

            if not png_files and not txt_files:
                empty_folders.append(subdir)
            elif not png_files:
                missing_png_folders.append(subdir)
            elif not txt_files:
                missing_txt_folders.append(subdir)
            else:
                for png_file in png_files:
                    file_path = os.path.join(subdir, png_file)
                    print(f'Processing {file_path}')
                    processed_text = process_png(file_path)

                    # 保存文字到图片所在的子文件夹中
                    text_file_path = os.path.join(subdir, 'output.txt')
                    save_text_to_file(processed_text, text_file_path)

    # 将结果保存到文件
    with open('missing_png_folders.txt', 'w') as file:
        for folder in missing_png_folders:
            file.write(folder + '\n')

    with open('missing_txt_folders.txt', 'w') as file:
        for folder in missing_txt_folders:
            file.write(folder + '\n')

    with open('empty_folders.txt', 'w') as file:
        for folder in empty_folders:
            file.write(folder + '\n')


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    KEY = config['DEFAULT']['API_KEY']
    client = OpenAI(api_key=KEY)
    MODEL = 'gpt-4o'
    pdf_path = "input/"
    output_path = "output/"
    process_pdfs_in_folder(pdf_path, output_path)
    output_dir = 'output'  # 指定output文件夹路径
    traverse_and_process_png(output_dir)
