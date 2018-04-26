
def get_valid_fct_args(fct, parser):

    fct_arguments = fct.__code__.co_varnames[:fct.__code__.co_argcount]
    parser_arguments = tuple(parser.__dict__.keys())
    common_args = [item for item in fct_arguments if item in parser_arguments]

    return {arg_name: getattr(parser, arg_name) for arg_name in common_args}