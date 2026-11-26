"""Incremental JSON-array input for signed, potentially giant source products."""
import gzip
import json


def array_items(path,chunk_size=1048576):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as handle:
        text='';position=0;eof=False;state='start';decoder=json.JSONDecoder()
        while True:
            while position<len(text) and text[position].isspace():position+=1
            if position==len(text) and not eof:
                text=text[position:]+handle.read(chunk_size);position=0;eof=len(text)==0
                continue
            if position==len(text):
                if state=='done':return
                raise ValueError('Truncated signed JSON array')
            token=text[position]
            if state=='start':
                if token!='[':raise ValueError('Expected a JSON array')
                position+=1;state='first';continue
            if state=='done':raise ValueError('Trailing content after JSON array')
            if state in ('first','separator') and token==']':position+=1;state='done';continue
            if state=='separator':
                if token!=',':raise ValueError('Missing JSON-array separator')
                position+=1;state='value';continue
            try:value,end=decoder.raw_decode(text,position)
            except json.JSONDecodeError:
                if eof:raise
                more=handle.read(chunk_size)
                if not more:eof=True
                text=text[position:]+more;position=0;continue
            # A number at a buffer boundary may be a prefix of a longer number.
            if not eof and (end==len(text) or (isinstance(value,(int,float)) and not isinstance(value,bool) and text[end] not in ',] \t\n\r')):
                more=handle.read(chunk_size)
                if more:text=text[position:]+more;position=0;continue
                eof=True
            position=end;state='separator';yield value
