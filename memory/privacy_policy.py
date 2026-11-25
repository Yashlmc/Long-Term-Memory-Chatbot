import re


class PrivacyPolicy:
    VALID_MODES = {"AUTO", "ASK", "DISABLED"}

    VALID_TYPES = {
        "profile",
        "preference",
        "interest",
        "skill",
        "project",
        "goal",
        "habit",
        "other",
    }

    # Obvious credentials, API tokens, and card/ID numbers
    SECRET_PATTERNS = [
        re.compile(r"\bpassword\s*(?:is|=|:)\s*\S+", re.IGNORECASE),
        re.compile(
            r"\b(?:api[_ -]?key|access[_ -]?token|secret[_ -]?key)\s*(?:is|=|:)\s*\S+",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:sk-[A-Za-z0-9_-]{10,}|gsk_[A-Za-z0-9_-]{10,})\b"),
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]+=*\b", re.IGNORECASE),
        # Unformatted or space/hyphen-separated 13-19 digit card numbers
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        # Indian Aadhaar (12 digits, optional spaces) & PAN format
        re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
        re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    ]

    # Sensitive personal/health/financial information in BOTH first-person
    # ("I have diabetes", "my bank account") AND third-person extracted form
    # ("User has diabetes", "User's passport number").
    SENSITIVE_PATTERNS = [
        re.compile(
            r"\b(?:my|i have|i was diagnosed with|i suffer from|"
            r"user(?:'s)?\s+(?:has|had|was diagnosed with|suffers from))\s+"
            r"(?:diagnosed with\s+)?"
            r"(?:diabetes|hiv|aids|cancer|depression|anxiety disorder|"
            r"bipolar disorder|schizophrenia|asthma|hypertension)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:my|user(?:'s)?)\s+(?:bank account|credit card|"
            r"debit card|passport number|aadhaar|aadhar|pan number|"
            r"government id|social security number|ssn)\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(self, mode="AUTO"):
        self.set_mode(mode)

    def set_mode(self, mode):
        mode = str(mode).strip().upper()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode: {mode}. Choose from {self.VALID_MODES}."
            )
        self.mode = mode

    def _validate_candidate(self, candidate):
        if not isinstance(candidate, dict):
            return False

        text = candidate.get("text")
        memory_type = candidate.get("type")
        importance = candidate.get("importance")
        confidence = candidate.get("confidence")

        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            return False

        if memory_type not in self.VALID_TYPES:
            return False

        if (
            isinstance(importance, bool)
            or not isinstance(importance, (int, float))
            or not 0 <= importance <= 1
        ):
            return False

        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            return False

        return True

    def _check_sensitivity(self, text):
        for pattern in self.SECRET_PATTERNS:
            if pattern.search(text):
                return "credential_or_secret"

        for pattern in self.SENSITIVE_PATTERNS:
            if pattern.search(text):
                return "sensitive_personal_information"

        return None

    def approve_candidate(self, candidate):
        if not self._validate_candidate(candidate):
            return False, "invalid_candidate"

        if self.mode == "DISABLED":
            return False, "memory_disabled"

        text = candidate["text"]
        reason = self._check_sensitivity(text)
        if reason:
            return False, reason

        if self.mode == "ASK":
            print("\nMemory candidate:")
            print(f"  Text: {text}")
            print(f"  Type: {candidate['type']}")

            answer = input("Store this memory? (y/n): ").strip().lower()
            if answer not in {"y", "yes"}:
                return False, "user_declined"

        return True, "approved"

    def filter_candidates(self, candidates):
        approved = []
        rejected = []

        for candidate in candidates:
            is_approved, reason = self.approve_candidate(candidate)
            if is_approved:
                approved.append(candidate)
            else:
                rejected.append({"candidate": candidate, "reason": reason})

        return {"approved": approved, "rejected": rejected}
