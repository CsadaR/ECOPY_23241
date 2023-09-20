#BRZIQA
def evens_from_list(input_list):
    even_values = []
    for item in input_list:
        if item % 2 == 0:
            even_values.append(item)
    return even_values

def every_element_is_odd(input_list):
    for item in input_list:
        if item % 2 != 0:
            return True
    return False

def kth_largest_in_list(input_list,kth_largest):
    input_list.sort()
    return input_list[-kth_largest]

def cumavg_list(input_list):
    cumulative_sum = 0
    cumulative_averages = []
    for i, num in enumerate(input_list, start=1):
        cumulative_sum += num
        average = cumulative_sum / i
        cumulative_averages.append(average)
    return cumulative_averages

def element_wise_multiplication(input_list1, input_list2):
	result = []
	for i in range(len(input_list1)):
		result.append(input_list1[i] * input_list2[i])
	return result

def merge_lists(*lists):
    merge_lists = []
    for l in lists:
        merge_lists.extend(l)
    return merge_lists

def squared_odds(input_list):
    odd_values = []
    for item in input_list:
        if item % 2 != 0:
            odd_values.append(item**2)
    return odd_values

def reverse_sort_by_key(input_dict):
    return dict(reversed(input_dict.items()))

def sort_list_by_divisibility(input_list):
    by_two = []
    by_five = []
    by_two_and_five = []
    by_none = []
    for num in input_list:
        if num % 2 == 0 and num % 5 == 0:by_two_and_five.append(num)
        elif num % 2 == 0:by_two.append(num)
        elif num % 5 == 0:by_five.append(num)
        else:by_none.append(num)
    result={ 'by_two': by_two,'by_five': by_five,'by_two_and_five': by_two_and_five,'by_none': by_none}
    return result