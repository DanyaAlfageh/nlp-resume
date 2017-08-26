import os
import json


def pdf_to_txt(root_dir):
    """ convert all pdfs in root_dir to text including sub dirs
    """
    for root, _, files in os.walk(root_dir):
        for f in files:
            _, ext = os.path.splitext(f)
            if ext == '.pdf':
                 os.system(('pdftotext "%s"') % os.path.join(root, f))


def txt_to_jsonl(root_dir, outpath):
    """ collect textfiles in root_dir and write them to jsonl
        remove textfiles after writing content to file
    """
    def file_path_iter(source_dir):
        """ keep only one file in memory at a time
        """
        return ((f, os.path.join(root, f))
                for root, dirs, files in os.walk(source_dir)
                for f in files if os.path.splitext(f)[1] == '.txt'
               )

    with open(outpath, 'w') as outf:
        for f, path in _file_path_iter(root_dir):
            with open(path, 'r', encoding='utf-8', errors='ignore') as inf:
                    json.dump({'name': f, 'content': inf.read()}, outf)
                    outf.write('\n')
            os.remove(path)


def process_pdfs(pdf_dir, out_path):
    """ wrap it all up
    """
    pdf_to_txt(pdf_dir)
    txt_to_jsonl(pdf_dir, out_path)

