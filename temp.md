## Here is how I originally used the code:

### From PDF to text

With all candidate pdf resumes in a dir I run 

``` python
pdftojsonl.process_pdfs('path to candidate dir', 'candidate.jsonl')
```

With all target resumes in a dir and run

``` python
pdftojsonl.process_pdfs('path to target dir', 'target.jsonl')
```

Now I have the two files `candidate.jsonl` and `target.jsonl` which contain all the pdf content but are easy to read / stream.

### Analysis

First I load the info from `candidate.jsonl` and `target.jsonl` into two lists:

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





