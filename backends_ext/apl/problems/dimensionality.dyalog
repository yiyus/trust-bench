Dimensionality←{
    n←⍺
    x←⍵
    pos←⍳n
    oddpos←(2|pos)/pos
    evenpos←(~2|pos)/pos
    a←x[oddpos]
    b←x[evenpos]
    r←n⍴0
    r[oddpos]←10×b-a×a
    r[evenpos]←1-a
    diagIdx←((oddpos-1)×n)+oddpos
    supIdx←((oddpos-1)×n)+evenpos
    subIdx←((evenpos-1)×n)+oddpos
    jflat←(n×n)⍴0
    jflat[diagIdx]←¯20×a
    jflat[supIdx]←(≢oddpos)⍴10.0
    jflat[subIdx]←(≢oddpos)⍴¯1.0
    j←n n⍴jflat
    r j
}
DimensionalityHessian←{
    n←⍺
    x←⍵
    r j←n Dimensionality x
    pos←⍳n
    oddpos←(2|pos)/pos
    diagIdx←((oddpos-1)×n)+oddpos
    hflat←,(⍉j)+.×j
    hflat[diagIdx]+←r[oddpos]×¯20
    n n⍴hflat
}
