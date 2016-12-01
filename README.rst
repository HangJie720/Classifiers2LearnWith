Classifiers2LearnWith
=====================
This is a library of classifiers put together by Andy Port for his own self-enrichment.

Database Formatting Standard
----------------------------
All databases are formatted in a similar way, as a MATLAB .mat file containing a tensor `X` containing sample points (indexed by the first dimension) and a column vector `y` containing labels.

To Get Databases
----------------
With the exception of small databases (e.g. the USPS database of handwritten digits), you must use the shell scripts `get_*.sh` to download (and format) each database.  Note: If you're on Windows, check out the `README.txt` files in each database's directory or just go through the steps in the `get_*.sh` files -- all these scripts do is: download the compressed databases, decompress/decode them, then run some python and/or octave scripts to reformat them to the above database formatting standard.

To Run the Classifiers
----------------------
Navigate to Classifiers2LearnWith/Classifiers directory, then run whichever you like.

Note on Matlab
--------------
When coding this I used Octave 4.2.0-rc2 ... I don't think you'll have any problems compiling the contained .m files with MATLAB, except that I have some suspicion that matlab does not have a built-in `argv()` function to grab CLI inputs.