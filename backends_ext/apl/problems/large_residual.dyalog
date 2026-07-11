LargeResidual←{
    a k←⍵
    m←40
    t←(¯1+⍳m)÷m-1
    y←(2.0×*1.3×t)+⍺×t-0.5
    e←*k×t
    r←(a×e)-y
    j←⍉2 m⍴e,a×t×e
    r j
}
LargeResidualHessian←{
    a k←⍵
    r j←⍺ LargeResidual ⍵
    m←40
    t←(¯1+⍳m)÷m-1
    e←*k×t
    h←(⍉j)+.×j
    s01←+/r×t×e
    s11←+/r×a×(t*2)×e
    h[1;2]+←s01 ⋄ h[2;1]+←s01 ⋄ h[2;2]+←s11
    h
}
