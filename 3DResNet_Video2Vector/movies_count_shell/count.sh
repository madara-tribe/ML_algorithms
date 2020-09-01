#!/bin/sh
for x in * ; do echo $x ; ls $x | wc -l ; done
# horizintal output
for x in * ; do echo -n "$x," ; ls $x | wc -l ; done
# hmdb51 rar　解凍
for i in *.rar; do echo $i ; unrar x $i; done


# any child dir file count
#!/bin/sh
for dir in *
do
  if [ -d "$dir" ]
  then
    echo "$dir"
    for subdir in "$dir"/*
    do
      if [ -d "$subdir" ]
      then
        echo "$dir"
        echo "${subdir##*/}" && echo "$(ls -1 "$subdir"|wc -l)"
      fi
    done
  fi
done


# horizon line output
#!/bin/sh
for dir in *
do
  if [ -d "$dir" ]
  then
    echo "$dir"
    for subdir in "$dir"/*
    do
      if [ -d "$subdir" ]
      then
        echo -n "$dir,"
        echo -n "${subdir##*/}", && echo "$(ls -1 "$subdir"|wc -l)"
      fi
    done
  fi
done
