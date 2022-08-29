for i in ptf_*.py
do
	sed '/time_tracker/d' "$i" > "$i"_temp
        mv "$i"_temp "$i"
done
