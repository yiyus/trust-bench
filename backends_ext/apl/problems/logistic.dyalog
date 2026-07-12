_LogisticX←0.01 0.03162277660168379 0.1 0.31622776601683794 1.0 3.1622776601683795 10.0 31.622776601683793 100.0
_LogisticY←1.0072021497369472 0.9024043733911785 0.14114128166335915 4.147876604418931 25.903351886031235 68.33996583548311 93.79042419717828 99.79681278978069 100.23780167593394
Logistic←{
    a b c d←⍵
    lr←⍟_LogisticX÷c
    ch←6○(b×lr)÷2
    u←*b×lr
    w←÷1+u
    da←w
    dd←1-w
    ch2←ch×ch
    amd←a-d
    db←(-lr×amd)÷4×ch2
    dc←(b×amd)÷4×c×ch2
    r←(d+amd÷1+u)-_LogisticY
    j←⍉4 9⍴da,db,dc,dd
    r j
}
LogisticHessian←{
    a b c d←⍵
    r j←Logistic ⍵
    lr←⍟_LogisticX÷c
    half←(b×lr)÷2
    ch←6○half
    sh←5○half
    ch2←ch×ch
    ch3←ch×ch2
    amd←a-d
    hab←(-lr)÷4×ch2
    hac←b÷4×c×ch2
    lr2←lr×lr
    hbb←(lr2×amd×sh)÷4×ch3
    inner←(b×lr×sh)-ch
    hbc←(-amd×inner)÷4×c×ch3
    hbd←lr÷4×ch2
    inner2←(b×sh)-ch
    hcc←(b×amd×inner2)÷4×c×c×ch3
    hcd←(-b)÷4×c×ch2
    h←(⍉j)+.×j
    c1←+/r×hab
    c2←+/r×hac
    c3←+/r×hbb
    c4←+/r×hbc
    c5←+/r×hbd
    c6←+/r×hcc
    c7←+/r×hcd
    h[1;2]+←c1 ⋄ h[2;1]+←c1
    h[1;3]+←c2 ⋄ h[3;1]+←c2
    h[2;2]+←c3
    h[2;3]+←c4 ⋄ h[3;2]+←c4
    h[2;4]+←c5 ⋄ h[4;2]+←c5
    h[3;3]+←c6
    h[3;4]+←c7 ⋄ h[4;3]+←c7
    h
}
