namespace SpiralV4

open FsUnit
open NUnit.Framework

module MnistTests =
    [<TestFixture>]
    type Testy() =
        let x = 1

        [<Test>]
        member __.a() = x |> should greaterThan 0


