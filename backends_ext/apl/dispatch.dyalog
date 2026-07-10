NameFor←{
    ⍵≡'rosenbrock':'Rosenbrock'
    ⍵≡'beale':'Beale'
    ⍵≡'powell':'Powell'
    ⍵≡'helical':'Helical'
    ⍵≡'expdec':'ExpDec'
    ⍵≡'quadratic':'Quadratic'
    ⍵≡'linear':'Linear'
    ''
}
Apply←{name point←⍵ ⋄ ⍎name,' point'}
