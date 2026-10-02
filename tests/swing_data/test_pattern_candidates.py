import numpy as np
import pandas as pd
from swing_data.pattern_candidates import detect_triggers as detect, prepare, pivots


def triangle_frame():
    n=101
    c=np.full(n,100.)
    x=np.arange(40)
    c[60:100]=100+(12-x*.2)*np.cos(x*np.pi/5)
    c[100]=112
    f=pd.DataFrame({'date':pd.bdate_range('2026-05-01',periods=n).strftime('%Y-%m-%d'),'adj_open':c-.5,'adj_close':c,'adj_high':c+1,'adj_low':c-1,'close':c,'volume':2_000_000.})
    f.loc[100,['adj_open','adj_low','adj_high','volume']]=[108,107,113,4_000_000]
    return f


def test_confirmed_pivots_do_not_include_last_two_bars():
    assert pivots(np.array([1.,2,3,2,1,4,9]),True).tolist()==[2]


def test_triangle_breakout_requires_volume_and_fresh_data():
    f=triangle_frame(); day=f.date.iloc[-1]
    matches=detect(f,day)
    assert any(x['kind']=='triangle_breakout' for x in matches)
    assert detect(f,'2026-10-02')==[]
    f.loc[100,'volume']=2_000_000
    assert detect(f,day)==[]


def test_invalid_ohlc_is_not_screened():
    f=triangle_frame(); f.loc[100,'adj_low']=120
    assert prepare(f,f.date.iloc[-1]) is None


def test_similarity_accepts_reference_and_rejects_opposite_shape():
    from swing_data.pattern_candidates import shape_similarity
    f=triangle_frame(); start=(pd.Timestamp(f.date.iloc[-1])-pd.Timedelta(days=120)).date().isoformat(); reference={'adj_close':f.loc[f.date>=start,'adj_close'].tolist()}
    assert shape_similarity(f,f.date.iloc[-1],reference)['matches']
    bad=f.copy()
    for k in ['adj_open','adj_high','adj_low','adj_close','close']: bad[k]=200-f[k]
    bad[['adj_high','adj_low']]=bad[['adj_low','adj_high']].to_numpy()
    assert not shape_similarity(bad,bad.date.iloc[-1],reference)['matches']
