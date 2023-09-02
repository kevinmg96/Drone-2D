def generic_function(arg1, arg2, arg3=False, *args, **kwargs):
    """
    A generic function with specified arguments and optional positional/keyword arguments.

    Parameters:
    arg1: Any
        The first required positional argument.
    arg2: Any
        The second required positional argument.
    arg3: bool, optional
        An optional third positional argument with a default value of False.
    *args: Tuple
        Additional positional arguments (variable-length).
    **kwargs: Dict
        Additional keyword arguments.

    Returns:
    str
        A formatted string combining the specified and additional arguments.
    """
    # Process the specified arguments
    specified_arguments = f"arg1={arg1}, arg2={arg2}, arg3={arg3}"

    # Process the additional positional arguments
    positional_arguments = ', '.join(map(str, args))

    # Process the additional keyword arguments
    keyword_arguments = ', '.join([f"{key}={value}" for key, value in kwargs.items()])

    # Combine all the arguments into a string
    result = f"Specified arguments: {specified_arguments}\nPositional arguments: {positional_arguments}\nKeyword arguments: {keyword_arguments}"

    return result

# Example usage
result = generic_function(1, 2, arg3=True, arg4="Hello", arg5=42)
print(result)



