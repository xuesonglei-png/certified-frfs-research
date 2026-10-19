from verify_mixture_envelope_symbolic import main


def test_exact_symbolic_example(capsys):
    main(); output=capsys.readouterr().out
    assert '9/10' in output and '1/2' in output
