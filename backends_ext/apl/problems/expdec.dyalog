ExpDec←{
    a k←⍵
    T←((⍳40)-1)÷39
    Y←2.0×*1.3×T
    e←*k×T
    r←(a×e)-Y
    J←⍉2 40⍴e,a×T×e
    r J
}
ExpDecHessian←{
    a k←⍵
    r j←ExpDec ⍵
    T←((⍳40)-1)÷39
    e←*k×T
    h←(⍉j)+.×j
    s01←+/r×T×e
    s11←+/r×a×T×T×e
    h+2 2⍴0 s01 s01 s11
}
