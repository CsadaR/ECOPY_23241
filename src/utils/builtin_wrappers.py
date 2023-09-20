def number_of_elements_in_list(input_list):
    return len(input_list)

def contains_value(input_list, element):
    return element in input_list

def sort_list_by_parity(input_list):
    result = {'even': [], 'odd': []}
    for num in input_list:
        result['even'].append(num) if num % 2 == 0 else result['odd'].append(num)
    return result