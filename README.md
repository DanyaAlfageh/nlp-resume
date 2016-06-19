## Document Relevance based on SGD Classifier and Cosine Similarity

This analysis entails determining how similar documents are to a target set of documents. The original problem involved narrowing down a large set of 'technical' resumes consisting of candidates from various fields such as IT, Business Analyst / Data Scientist, and Software Engineering to a subset of resumes most relevant for an Analyst position.

To solve this problem, resumes of employees who already fill technical positions are collected and labeled.  Then the set of candidate resumes is compared for relevance to the labeled resumes.

The end result does not assess how well a candidate may perform at a particular position but should associate people with relevant resumes to the target position.

### Python Environment
Using Conda, Anaconda Python 3.5.1 :: Continuum Analytics, Inc. with requirements listed in [environment.yml](https://github.com/blakeboswell/nlp-resume/blob/master/environment.yml)

Create environment via:
``` bash
# from project root
conda env create
```

### Data Preparation

The resumes originate in pdf form (already having ocr). Conversion to raw text is performed using the `pdftotext` utility.
- Instructions for installation on OSX can be found [here](http://macappstore.org/pdftotext/).
- Download information for Windows can be found [here](http://www.foolabs.com/xpdf/download.html)

### Scratch Work

All the scratch work for testing the approach is shown in this [notebook](https://github.com/blakeboswell/nlp-resume/blob/master/newsgroup_test.ipynb).  The newsgroup data is used as a substitute for resumes.


## Relating back to Resume Analysis

### Data Preparation

With all candidate pdf resumes in a dir run

``` python
pdftojsonl.process_pdfs('path/to/candidate/dir', 'candidate.jsonl')
```

With all target resumes in a dir and run

``` python
pdftojsonl.process_pdfs('path/to/target/dir', 'target.jsonl')
```

`candidate.jsonl` and `target.jsonl` now exist and contain all the pdf content but are easy to read / stream.

### Analysis

Load the info from `candidate.jsonl` and `target.jsonl` into two lists:

``` python
import json
with open('candidate.jsonl', 'r') as cf, open('target.jsonl', 'r') as tf:
  candidate = [json.loads(line) for line in cf]
  target = [json.loads(line) for line in tf]
```

So now I have `target` and `candidate`.  However, for this analysis I concatenate all the targets together into one:

``` python
mashed_target = ''
for t in target:
  mashed_target += t['content']
target = [{'name':'target', 'content': mashed_target}]
```

So now I have `target` which is a list conaining one dictionary with all the target resume's content. So i can calculate the similarity as:

``` python
import main
d = main.similarity(target, candidate)
```
