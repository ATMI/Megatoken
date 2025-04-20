def prompt(text: str) -> bool:
	while True:
		response = input(text).strip().lower()
		if response in ["yes", "y"]:
			return True
		elif response in ["no", "n"]:
			return False
		else:
			print("Please answer with 'yes' or 'no'.")
