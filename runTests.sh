#!/usr/bin/env bash 
echo "Тест №1 - Проверка синтаксиса"
python Lab3.py >> out
if $?
then
    echo "Тест провален!"
    exit 1
else
    echo "Тест пройден. Нет синтаксических ошибок."
fi
echo "Тест №2 - Проверка коэффициентов"
var=$(cat out | grep Вариант | grep -o -E '[0-9]+')
B0=$(grep 'B\[0\]' out)
B1=$(grep 'B\[1\]' out)
B2=$(grep 'B\[2\]' out)
B3=$(grep 'B\[3\]' out)
checkB0=$(awk -F: '{if ($1 == "$var") {print $2}}' ./tests/res_koef.txt)
checkB1=$(awk -F: '{if ($1 == "$var") {print $3}}' ./tests/res_koef.txt )
checkB2=$(awk -F: '{if ($1 == "$var") {print $4}}' ./tests/res_koef.txt )
checkB3=$(awk -F: '{if ($1 == "$var") {print $5}}' ./tests/res_koef.txt )
if [[ "$B0" == *"$checkB0"* ]]
then
    echo "B[0] совпадает"
else
    echo "B[0] не совпадает"
    echo "Тест провален!"
    exit 1
fi
if [[ "$B1" == *"$checkB1"* ]]
then
    echo "B[1] совпадает"
else
    echo "B[1] не совпадает"
    echo "Тест провален!"
    exit 1
fi
if [[ "$B2" == *"$checkB2"* ]]
then
    echo "B[2] совпадает"
else
    echo "B[2] не совпадает"
    echo "Тест провален!"
    exit 1
fi
if [[ "$B3" == *"$checkB3"* ]]
then
    echo "B[3] совпадает"
else
    echo "B[3] не совпадает"
    echo "Тест провален!"
    exit 1
fi
echo "Тест пройден!"
exit 0