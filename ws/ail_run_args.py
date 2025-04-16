from ail_parser import parse_intermixed_args

# output the parsed arguments in a format that can be consumed by bash
if __name__ == "__main__":
    args, rest = parse_intermixed_args(uninstalled_requirements=True)
    print("ail_opt_successful_arg_parse=1")
    for arg in vars(args):
        value = getattr(args, arg)
        formatted_value = +value if isinstance(value, bool) else value
        print(f"ail_opt_{arg}={formatted_value}")
