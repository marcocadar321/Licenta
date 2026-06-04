

#include <cstring>
#include <cctype>
#include <cstdlib>
#include <string>
#include <sstream>


#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

extern "C" {


EXPORT char* normalize_text(const char* input) {
    if (!input) return nullptr;
    std::string s(input);
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
        if (std::isalnum((unsigned char)c) || c == ' ') {
            result += std::tolower((unsigned char)c);
        } else {
            
            result += ' ';
        }
    }
  
    std::string clean;
    clean.reserve(result.size());
    bool lastSpace = false;
    for (char c : result) {
        if (c == ' ') {
            if (!lastSpace && !clean.empty()) clean += ' ';
            lastSpace = true;
        } else {
            clean += c;
            lastSpace = false;
        }
    }
    
    while (!clean.empty() && clean.back() == ' ') clean.pop_back();

    char* out = (char*)malloc(clean.size() + 1);
    if (out) std::memcpy(out, clean.c_str(), clean.size() + 1);
    return out;
}


EXPORT char* tokenize(const char* input) {
    if (!input) return nullptr;
    std::istringstream iss(input);
    std::string token;
    std::string result;
    while (iss >> token) {
        if (!result.empty()) result += '\n';
        result += token;
    }
    char* out = (char*)malloc(result.size() + 1);
    if (out) std::memcpy(out, result.c_str(), result.size() + 1);
    return out;
}


EXPORT int count_tokens(const char* input) {
    if (!input) return 0;
    std::istringstream iss(input);
    std::string token;
    int count = 0;
    while (iss >> token) ++count;
    return count;
}


EXPORT void free_string(char* ptr) {
    free(ptr);
}

} 
