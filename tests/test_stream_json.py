import gzip,json
import pytest
from deep_pm.stream_json import array_items


def test_signed_source_array_streams_nested_and_unicode_values_over_chunk_edges(tmp_path):
    rows=[{'ids':['quote"','new\nline','恒星']*20,'v':-12.345e-12,'missing':None},{'nested':[[1,2],[3,4]],'w':[False,True]}]
    for name in ('source.json','source.json.gz'):
        path=tmp_path/name;opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'wt',encoding='utf-8') as f:json.dump(rows,f,ensure_ascii=False)
        assert list(array_items(path,7))==rows


@pytest.mark.parametrize('value',['[1,]','[1','[]false','{"x":1}','[1 2]'])
def test_corrupt_or_nonarray_input_is_rejected(tmp_path,value):
    p=tmp_path/'bad.json';p.write_text(value)
    with pytest.raises((ValueError,json.JSONDecodeError)):list(array_items(p,2))


def test_array_number_and_empty_edges(tmp_path):
    p=tmp_path/'values.json';p.write_text('[12345, -3.121e-12, [], {}, ""]')
    assert list(array_items(p,2))==json.loads(p.read_text())
