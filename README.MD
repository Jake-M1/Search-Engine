Search Engine Instructions
Jake Mulligan

How to Run Search Engine and Type in a Query:
- Run the launch.py script
- Go to http://localhost:5000 in a web browser
- Type in a query in the search bar
- Click submit or hit enter on the keyboard
- Displayed are the 10 most relevent urls

How to Index:
- Uncomment the 2nd line in the main function of launch.py (indexer.build_index())
- Run launch.py
- This will create many files, like partial indexes, so the main file is called index.txt

How to Time / Not Time Query Calculations:
- Change the timing variable in launch.py (5th line of main) to either True or False
- Run launch.py
- Go to http://localhost:5000 in a web browser
- Type in a query in the search bar
- Click submit or hit enter on the keyboard
- The time to process and display the results is displayed in the terminal
