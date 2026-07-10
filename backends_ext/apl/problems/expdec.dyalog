ExpDec←{
    a k←⍵
    T←((⍳40)-1)÷39
    Y←2.0×*1.3×T
    e←*k×T
    r←(a×e)-Y
    J←⍉2 40⍴e,a×T×e
    r J
}
