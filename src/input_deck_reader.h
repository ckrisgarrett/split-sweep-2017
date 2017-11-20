/*
    File:   input_deck_reader.h
    Author: Kris Garrett
    Date:   September 3, 2013
*/

#ifndef __INPUT_DECK_READER_H
#define __INPUT_DECK_READER_H

#include <list>

class InputDeckReader {
public:
    static const int VALUE_CHAR_LEN = 128;
    
    void readInputDeck(const char *filename);
    bool getValue(const char *key, char value[VALUE_CHAR_LEN]);
    bool getValue(const char *key, int *value);
    bool getValue(const char *key, double *value);
    bool getValue(const char *key, bool *value);
    void print();
    
private:
    struct KeyValue {
        char key[VALUE_CHAR_LEN];
        char value[VALUE_CHAR_LEN];
    };
    
    std::list<KeyValue> keyValueList;
};

#endif
