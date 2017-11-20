/*
    File:   input_deck_reader.cpp
    Author: Kris Garrett
    Date:   September 3, 2013
*/

#include "input_deck_reader.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>


/*
    Reads the input deck from a file.
*/
void InputDeckReader::readInputDeck(const char *filename)
{
    // Variables
    const int MAX_LINE_LENGTH = 500;
    char line[MAX_LINE_LENGTH];
    FILE *file;
    int lineNumber;
    KeyValue kv;
    char *ret;
    char *newline;
    char *comment;
    char *pch;
    
    
    // Try to open the file.
    file = fopen(filename, "r");
    if(file == NULL) {
        printf("Could not open file.\n");
        return;
    }
    
    
    // Go through input line by line.
    lineNumber = 0;
    while(true) {
        lineNumber++;
        
        // Get line.
        ret = fgets(line, MAX_LINE_LENGTH, file);
        if(ret == NULL)
            break;
        
        // Get rid of newline character.
        newline = strchr(line, '\n');
        if(newline != NULL)
            newline[0] = 0;
        
        // Get rid of comment from line.
        comment = strchr(line, '#');
        if(comment != NULL)
            comment[0] = 0;
        
        // Tokenize the line by spaces.
        pch = strtok(line, " \t");
        if(pch != NULL)
            strncpy(kv.key, pch, VALUE_CHAR_LEN);
        else
            continue;
        pch = strtok(NULL, " ");
        if(pch != NULL)
            strncpy(kv.value, pch, VALUE_CHAR_LEN);
        else
            printf("Error in input file at line: %d\n", lineNumber);
        pch = strtok(NULL, " ");
        if(pch == NULL)
            keyValueList.push_back(kv);
        else
            printf("Error in input file at line: %d\n", lineNumber);
    }
    
    fclose(file);
}


/*
    Helper functions for making strings lowercase.
*/
static
void makeLowercase(const char *input, char *output)
{
    int i;
    for(i = 0; input[i] != 0; i++)
        output[i] = tolower(input[i]);
    output[i] = 0;
}

static
void makeLowercase(char *s)
{
    for(int i = 0; s[i] != 0; i++)
        s[i] = tolower(s[i]);
}


/*
    Gets a value given the key.
*/
bool InputDeckReader::getValue(const char *key, char value[InputDeckReader::VALUE_CHAR_LEN])
{
    char keyLowercase1[InputDeckReader::VALUE_CHAR_LEN];
    char keyLowercase2[InputDeckReader::VALUE_CHAR_LEN];
    
    for(std::list<KeyValue>::iterator it=keyValueList.begin(); it!=keyValueList.end(); ++it) {
        makeLowercase(it->key, keyLowercase1);
        makeLowercase(key, keyLowercase2);
        if(strcmp(keyLowercase1, keyLowercase2) == 0) {
            strcpy(value, it->value);
            makeLowercase(value);
            return true;
        }
    }
    
    return false;
}


bool InputDeckReader::getValue(const char *key, int *value)
{
    char valueString[InputDeckReader::VALUE_CHAR_LEN];
    
    if(getValue(key, valueString)) {
        *value = atoi(valueString);
        return true;
    }
    return false;
}


bool InputDeckReader::getValue(const char *key, double *value)
{
    char valueString[InputDeckReader::VALUE_CHAR_LEN];
    
    if(getValue(key, valueString)) {
        *value = atof(valueString);
        return true;
    }
    return false;
}


bool InputDeckReader::getValue(const char *key, bool *value)
{
    char valueString[InputDeckReader::VALUE_CHAR_LEN];
    
    if(getValue(key, valueString)) {
        if(strcmp(valueString, "true") == 0)
            *value = true;
        else
            *value = false;
        return true;
    }
    return false;
}


/*
    Prints the entire input deck.
*/
void InputDeckReader::print()
{
    printf("--- Input Deck ---\n");
    for(std::list<KeyValue>::iterator it=keyValueList.begin(); it!=keyValueList.end(); ++it) {
        printf("%s = %s\n", it->key, it->value);
    }
}
