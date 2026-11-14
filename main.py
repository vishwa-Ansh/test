"""
College Management API (single file) - FastAPI + PostgreSQL
Roles : admin, teacher, student
Modules: auth, profiles, face attendance, manual attendance, scores, performance, results

INSTALL
  pip install fastapi "uvicorn[standard]" sqlalchemy psycopg2-binary "pydantic[email]" \
              pyjwt bcrypt python-multipart pillow numpy
  pip install face_recognition        # needs cmake + dlib (face features ke liye)

ENV VARIABLES
  DATABASE_URL    postgresql+psycopg2://user:pass@localhost:5432/college
  SECRET_KEY      long random string (production me zaroor set karo)
  ADMIN_EMAIL     first admin email   (default: admin@college.local)
  ADMIN_PASSWORD  first admin password (set nahi kiya to admin seed nahi hoga)
  CORS_ORIGINS    comma separated origins 
"""
import enum
import io
import logging
import os
import re
import secrets
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal, Optional

import bcrypt
import jwt
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi import status as st
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import AfterValidator, BaseModel, ConfigDict, EmailStr, Field, model_validator
from sqlalchemy import (ARRAY, Boolean, Date, DateTime, Float, ForeignKey, Integer, String,
                        Text, UniqueConstraint, case, create_engine, func, or_, select)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

try:
    import face_recognition
    FACE_OK = True
except ImportError:  # app chalega, sirf face endpoints 503 denge
    FACE_OK = False

log = logging.getLogger("college")
logging.basicConfig(level=logging.INFO)

# ───────────────────────── Config ─────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/college")
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(32)
    log.warning("SECRET_KEY set nahi hai: random key use ho rahi hai (restart par tokens invalid honge)")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MIN = 480
FACE_TOLERANCE = 0.5          # kam = strict matching
MAX_IMAGE_BYTES = 5 * 1024 * 1024
PASS_PCT = 40.0
MIN_ATTENDANCE_PCT = 75.0
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
Image.MAX_IMAGE_PIXELS = 40_000_000

# ───────────────────────── Database ─────────────────────────
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Role(str, enum.Enum):
    admin = "admin"
    teacher = "teacher"
    student = "student"


class AttStatus(str, enum.Enum):
    present = "present"
    absent = "absent"
    late = "late"


class AttMethod(str, enum.Enum):
    face = "face"
    manual = "manual"


class ExamType(str, enum.Enum):
    quiz = "quiz"
    assignment = "assignment"
    midterm = "midterm"
    practical = "practical"
    final = "final"


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(120))
    phone: Mapped[Optional[str]] = mapped_column(String(15))
    role: Mapped[Role] = mapped_column(SAEnum(Role, name="role"), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    student: Mapped[Optional["Student"]] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")
    teacher: Mapped[Optional["Teacher"]] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")


class Student(Base):
    __tablename__ = "students"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    roll_no: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    department: Mapped[str] = mapped_column(String(80), index=True)
    semester: Mapped[int] = mapped_column(Integer)
    section: Mapped[Optional[str]] = mapped_column(String(5))
    dob: Mapped[date] = mapped_column(Date)
    gender: Mapped[str] = mapped_column(String(10))
    address: Mapped[Optional[str]] = mapped_column(Text)
    guardian_name: Mapped[Optional[str]] = mapped_column(String(120))
    guardian_phone: Mapped[Optional[str]] = mapped_column(String(15))
    photo_path: Mapped[Optional[str]] = mapped_column(String(255))
    face_encoding: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Float), nullable=True)
    user: Mapped[User] = relationship(back_populates="student")


class Teacher(Base):
    __tablename__ = "teachers"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    employee_id: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    department: Mapped[str] = mapped_column(String(80), index=True)
    designation: Mapped[Optional[str]] = mapped_column(String(80))
    qualification: Mapped[Optional[str]] = mapped_column(String(120))
    user: Mapped[User] = relationship(back_populates="teacher")


class Subject(Base):
    __tablename__ = "subjects"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    department: Mapped[str] = mapped_column(String(80), index=True)
    semester: Mapped[int] = mapped_column(Integer)
    teacher_id: Mapped[Optional[int]] = mapped_column(ForeignKey("teachers.id", ondelete="SET NULL"))


class Attendance(Base):
    __tablename__ = "attendance"
    __table_args__ = (UniqueConstraint("student_id", "subject_id", "date", name="uq_attendance_day"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), index=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    status: Mapped[AttStatus] = mapped_column(SAEnum(AttStatus, name="att_status"))
    method: Mapped[AttMethod] = mapped_column(SAEnum(AttMethod, name="att_method"))
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    marked_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class Score(Base):
    __tablename__ = "scores"
    __table_args__ = (UniqueConstraint("student_id", "subject_id", "exam_name", name="uq_score_exam"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), index=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    exam_type: Mapped[ExamType] = mapped_column(SAEnum(ExamType, name="exam_type"))
    exam_name: Mapped[str] = mapped_column(String(60))
    marks_obtained: Mapped[float] = mapped_column(Float)
    max_marks: Mapped[float] = mapped_column(Float)
    exam_date: Mapped[date] = mapped_column(Date)
    remarks: Mapped[Optional[str]] = mapped_column(String(255))
    recorded_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DB = Annotated[Session, Depends(get_db)]

# ───────────────────────── Validation types ─────────────────────────
def _password(v: str) -> str:
    if not 8 <= len(v) <= 64:
        raise ValueError("Password 8 se 64 characters ka hona chahiye")
    if not (re.search(r"[a-z]", v) and re.search(r"[A-Z]", v) and re.search(r"\d", v)):
        raise ValueError("Password me uppercase, lowercase aur digit hona chahiye")
    return v


Password = Annotated[str, AfterValidator(_password)]
Phone = Annotated[str, Field(pattern=r"^[6-9]\d{9}$")]
Name = Annotated[str, Field(min_length=2, max_length=120)]
Dept = Annotated[str, Field(min_length=2, max_length=80), AfterValidator(str.upper)]
Code = Annotated[str, Field(pattern=r"^[A-Za-z0-9\-]{3,20}$"), AfterValidator(str.upper)]
Semester = Annotated[int, Field(ge=1, le=8)]


class Strict(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class LoginIn(Strict):
    email: EmailStr
    password: str = Field(min_length=1, max_length=64)
    role: Role


class ChangePasswordIn(Strict):
    old_password: str = Field(min_length=1, max_length=64)
    new_password: Password


class StudentCreate(Strict):
    email: EmailStr
    password: Password
    full_name: Name
    phone: Optional[Phone] = None
    roll_no: Code
    department: Dept
    semester: Semester
    section: Optional[str] = Field(None, max_length=5)
    dob: date
    gender: Literal["male", "female", "other"]
    address: Optional[str] = Field(None, max_length=500)
    guardian_name: Optional[Name] = None
    guardian_phone: Optional[Phone] = None

    @model_validator(mode="after")
    def check_dob(self):
        age = (date.today() - self.dob).days / 365.25
        if not 15 <= age <= 60:
            raise ValueError("Student ki age 15 se 60 saal ke beech honi chahiye")
        return self


class StudentUpdate(Strict):
    full_name: Optional[Name] = None
    phone: Optional[Phone] = None
    department: Optional[Dept] = None
    semester: Optional[Semester] = None
    section: Optional[str] = Field(None, max_length=5)
    address: Optional[str] = Field(None, max_length=500)
    guardian_name: Optional[Name] = None
    guardian_phone: Optional[Phone] = None


class TeacherCreate(Strict):
    email: EmailStr
    password: Password
    full_name: Name
    phone: Optional[Phone] = None
    employee_id: Code
    department: Dept
    designation: Optional[str] = Field(None, max_length=80)
    qualification: Optional[str] = Field(None, max_length=120)


class SubjectCreate(Strict):
    code: Code
    name: Name
    department: Dept
    semester: Semester
    teacher_id: Optional[int] = Field(None, ge=1)


class StatusIn(Strict):
    is_active: bool


class AttendanceItem(Strict):
    student_id: int = Field(ge=1)
    status: AttStatus


class ManualAttendanceIn(Strict):
    subject_id: int = Field(ge=1)
    attendance_date: date
    records: list[AttendanceItem] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def check(self):
        if self.attendance_date > date.today():
            raise ValueError("Future date ki attendance nahi lag sakti")
        ids = [r.student_id for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Records me duplicate student_id hai")
        return self


class ScoreIn(Strict):
    student_id: int = Field(ge=1)
    subject_id: int = Field(ge=1)
    exam_type: ExamType
    exam_name: str = Field(min_length=2, max_length=60)
    marks_obtained: float = Field(ge=0)
    max_marks: float = Field(gt=0)
    exam_date: date
    remarks: Optional[str] = Field(None, max_length=255)

    @model_validator(mode="after")
    def check(self):
        if self.marks_obtained > self.max_marks:
            raise ValueError("marks_obtained, max_marks se zyada nahi ho sakte")
        if self.exam_date > date.today():
            raise ValueError("exam_date future me nahi ho sakti")
        return self


class ScoreUpdate(Strict):
    marks_obtained: Optional[float] = Field(None, ge=0)
    max_marks: Optional[float] = Field(None, gt=0)
    remarks: Optional[str] = Field(None, max_length=255)

# ───────────────────────── Security ─────────────────────────
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/token")


def hash_password(p: str) -> str:
    return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()


def verify_password(p: str, h: str) -> bool:
    try:
        return bcrypt.checkpw(p.encode(), h.encode())
    except ValueError:
        return False


def create_token(user: User) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MIN)
    return jwt.encode({"sub": str(user.id), "role": user.role.value, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)


def authenticate(db: Session, email: str, password: str) -> User:
    user = db.scalar(select(User).where(User.email == email.lower()))
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(st.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    if not user.is_active:
        raise HTTPException(st.HTTP_403_FORBIDDEN, "Account disabled hai")
    return user


def get_current_user(token: Annotated[str, Depends(oauth2)], db: DB) -> User:
    err = HTTPException(st.HTTP_401_UNAUTHORIZED, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = db.get(User, int(payload["sub"]))
    except (jwt.PyJWTError, KeyError, ValueError):
        raise err
    if not user or not user.is_active:
        raise err
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_roles(*roles: Role):
    def checker(user: CurrentUser) -> User:
        if user.role not in roles:
            raise HTTPException(st.HTTP_403_FORBIDDEN, "Is action ki permission nahi hai")
        return user
    return checker


AdminOnly = Annotated[User, Depends(require_roles(Role.admin))]
StaffOnly = Annotated[User, Depends(require_roles(Role.admin, Role.teacher))]
StudentOnly = Annotated[User, Depends(require_roles(Role.student))]

# ───────────────────────── Helpers ─────────────────────────
def create_user(db: Session, *, email: str, password: str, full_name: str, phone: Optional[str], role: Role) -> User:
    email = email.lower()
    if db.scalar(select(User.id).where(User.email == email)):
        raise HTTPException(st.HTTP_409_CONFLICT, "Email already registered")
    user = User(email=email, password_hash=hash_password(password), full_name=full_name, phone=phone, role=role)
    db.add(user)
    db.flush()
    return user


def student_out(s: Student) -> dict:
    u = s.user
    return {
        "id": s.id, "user_id": u.id, "role": "student", "email": u.email, "full_name": u.full_name,
        "phone": u.phone, "is_active": u.is_active, "roll_no": s.roll_no, "department": s.department,
        "semester": s.semester, "section": s.section, "dob": s.dob, "gender": s.gender,
        "address": s.address, "guardian_name": s.guardian_name, "guardian_phone": s.guardian_phone,
        "face_enrolled": s.face_encoding is not None,
    }


def teacher_out(t: Teacher) -> dict:
    u = t.user
    return {
        "id": t.id, "user_id": u.id, "role": "teacher", "email": u.email, "full_name": u.full_name,
        "phone": u.phone, "is_active": u.is_active, "employee_id": t.employee_id,
        "department": t.department, "designation": t.designation, "qualification": t.qualification,
    }


def user_out(u: User) -> dict:
    if u.role == Role.student and u.student:
        return student_out(u.student)
    if u.role == Role.teacher and u.teacher:
        return teacher_out(u.teacher)
    return {"id": u.id, "user_id": u.id, "role": u.role.value, "email": u.email,
            "full_name": u.full_name, "phone": u.phone, "is_active": u.is_active}


def my_student(db: Session, user: User) -> Student:
    s = db.scalar(select(Student).where(Student.user_id == user.id))
    if not s:
        raise HTTPException(st.HTTP_404_NOT_FOUND, "Student profile nahi mila")
    return s


def student_for(db: Session, user: User, student_id: int) -> Student:
    s = db.get(Student, student_id)
    if not s:
        raise HTTPException(st.HTTP_404_NOT_FOUND, "Student not found")
    if user.role == Role.student and s.user_id != user.id:
        raise HTTPException(st.HTTP_403_FORBIDDEN, "Aap sirf apna data dekh sakte ho")
    return s


def subject_for(db: Session, user: User, subject_id: int) -> Subject:
    sub = db.get(Subject, subject_id)
    if not sub:
        raise HTTPException(st.HTTP_404_NOT_FOUND, "Subject not found")
    if user.role == Role.teacher:
        t = db.scalar(select(Teacher).where(Teacher.user_id == user.id))
        if not t or sub.teacher_id != t.id:
            raise HTTPException(st.HTTP_403_FORBIDDEN, "Ye subject aapko assign nahi hai")
    return sub


def pct(a: float, b: float) -> Optional[float]:
    return round(100 * a / b, 2) if b else None


def grade(p: float) -> str:
    for cut, g in ((90, "O"), (80, "A+"), (70, "A"), (60, "B+"), (50, "B"), (40, "C")):
        if p >= cut:
            return g
    return "F"


def load_image(upload: UploadFile) -> Image.Image:
    if upload.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(st.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Sirf JPEG ya PNG image allowed hai")
    raw = upload.file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(st.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Image {MAX_IMAGE_BYTES // 1024 // 1024}MB se badi hai")
    try:
        Image.open(io.BytesIO(raw)).verify()
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(st.HTTP_400_BAD_REQUEST, "Valid image nahi hai")
    img.thumbnail((1600, 1600))
    return img


def need_face_lib():
    if not FACE_OK:
        raise HTTPException(st.HTTP_503_SERVICE_UNAVAILABLE, "face_recognition library install nahi hai")


def upsert_attendance(db: Session, existing: dict, student_id: int, subject_id: int, d: date,
                      status_: AttStatus, method: AttMethod, marker: int, conf: Optional[float] = None):
    rec = existing.get(student_id)
    if rec:
        rec.status, rec.method, rec.marked_by, rec.confidence = status_, method, marker, conf
    else:
        rec = Attendance(student_id=student_id, subject_id=subject_id, date=d, status=status_,
                         method=method, marked_by=marker, confidence=conf)
        db.add(rec)
        existing[student_id] = rec


def existing_attendance(db: Session, subject_id: int, d: date) -> dict:
    rows = db.scalars(select(Attendance).where(Attendance.subject_id == subject_id, Attendance.date == d))
    return {a.student_id: a for a in rows}


def subject_stats(db: Session, s: Student, semester: int) -> list[dict]:
    subjects = db.scalars(select(Subject).where(Subject.department == s.department, Subject.semester == semester)
                          .order_by(Subject.code)).all()
    attended = func.coalesce(func.sum(case((Attendance.status.in_([AttStatus.present, AttStatus.late]), 1), else_=0)), 0)
    att = {r[0]: (r[1], int(r[2])) for r in db.execute(
        select(Attendance.subject_id, func.count(Attendance.id), attended)
        .where(Attendance.student_id == s.id).group_by(Attendance.subject_id))}
    sc = {r[0]: (float(r[1]), float(r[2])) for r in db.execute(
        select(Score.subject_id, func.sum(Score.marks_obtained), func.sum(Score.max_marks))
        .where(Score.student_id == s.id).group_by(Score.subject_id))}
    rows = []
    for sub in subjects:
        held, got_att = att.get(sub.id, (0, 0))
        got, mx = sc.get(sub.id, (0.0, 0.0))
        rows.append({"subject_id": sub.id, "code": sub.code, "name": sub.name,
                     "classes_held": held, "classes_attended": got_att, "attendance_pct": pct(got_att, held),
                     "marks_obtained": got, "max_marks": mx, "score_pct": pct(got, mx)})
    return rows


def performance_for(db: Session, s: Student) -> dict:
    rows = subject_stats(db, s, s.semester)
    held = sum(r["classes_held"] for r in rows)
    attended = sum(r["classes_attended"] for r in rows)
    got = sum(r["marks_obtained"] for r in rows)
    mx = sum(r["max_marks"] for r in rows)
    att_pct, score_pct = pct(attended, held), pct(got, mx)
    warnings = []
    for r in rows:
        if r["attendance_pct"] is not None and r["attendance_pct"] < MIN_ATTENDANCE_PCT:
            warnings.append(f"Low attendance in {r['code']} ({r['attendance_pct']}%)")
        if r["score_pct"] is not None and r["score_pct"] < PASS_PCT:
            warnings.append(f"Below pass marks in {r['code']} ({r['score_pct']}%)")
    if score_pct is None:
        level = "No data"
    elif score_pct >= 75 and (att_pct or 0) >= MIN_ATTENDANCE_PCT:
        level = "Excellent"
    elif score_pct >= 60:
        level = "Good"
    elif score_pct >= PASS_PCT:
        level = "Average"
    else:
        level = "Needs improvement"
    trend = db.execute(
        select(Score.exam_date, Score.exam_name, Subject.code, Score.marks_obtained, Score.max_marks)
        .join(Subject, Subject.id == Score.subject_id).where(Score.student_id == s.id)
        .order_by(Score.exam_date.desc()).limit(10)).all()
    return {
        "student": {"id": s.id, "name": s.user.full_name, "roll_no": s.roll_no,
                    "department": s.department, "semester": s.semester},
        "overall_attendance_pct": att_pct, "overall_score_pct": score_pct, "performance_level": level,
        "warnings": warnings, "subjects": rows,
        "recent_scores": [{"date": t[0], "exam": t[1], "subject": t[2], "marks": t[3], "max": t[4],
                           "pct": pct(t[3], t[4])} for t in reversed(trend)],
    }


def result_for(db: Session, s: Student, semester: Optional[int]) -> dict:
    sem = semester or s.semester
    rows = subject_stats(db, s, sem)
    out, declared_got, declared_max, all_declared, any_fail = [], 0.0, 0.0, True, False
    for r in rows:
        p = r["score_pct"]
        if p is None:
            all_declared = False
            out.append({**r, "grade": None, "result": "NOT DECLARED"})
            continue
        declared_got += r["marks_obtained"]
        declared_max += r["max_marks"]
        failed = p < PASS_PCT
        any_fail = any_fail or failed
        out.append({**r, "grade": grade(p), "result": "FAIL" if failed else "PASS"})
    overall = pct(declared_got, declared_max)
    if not rows:
        final = "NO SUBJECTS"
    elif not all_declared:
        final = "INCOMPLETE"
    else:
        final = "FAIL" if any_fail else "PASS"
    return {"student": {"id": s.id, "name": s.user.full_name, "roll_no": s.roll_no, "department": s.department},
            "semester": sem, "subjects": out, "total_obtained": declared_got, "total_max": declared_max,
            "percentage": overall, "overall_grade": grade(overall) if overall is not None else None,
            "final_result": final}

# ───────────────────────── App ─────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    (UPLOAD_DIR / "faces").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    admin_pw = os.getenv("ADMIN_PASSWORD")
    with SessionLocal() as db:
        if db.scalar(select(User.id).where(User.role == Role.admin)) is None:
            if admin_pw:
                create_user(db, email=os.getenv("ADMIN_EMAIL", "admin@college.local"), password=admin_pw,
                            full_name="System Admin", phone=None, role=Role.admin)
                db.commit()
                log.info("First admin create ho gaya")
            else:
                log.warning("Koi admin nahi hai: ADMIN_PASSWORD set karke restart karo")
    yield


app = FastAPI(title="College Management API", version="1.0.0", lifespan=lifespan)
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=origins != ["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "face_recognition": FACE_OK}

# ───────────────────────── Auth ─────────────────────────
@app.post("/auth/login", tags=["Auth"])
def login(data: LoginIn, db: DB):
    user = authenticate(db, data.email, data.password)
    if user.role != data.role:
        raise HTTPException(st.HTTP_403_FORBIDDEN, f"Ye {data.role.value} account nahi hai")
    return {"access_token": create_token(user), "token_type": "bearer", "role": user.role.value, "user": user_out(user)}


@app.post("/auth/token", tags=["Auth"], include_in_schema=True)
def token(form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DB):
    """Swagger 'Authorize' button ke liye (username = email)."""
    user = authenticate(db, form.username, form.password)
    return {"access_token": create_token(user), "token_type": "bearer"}


@app.get("/auth/me", tags=["Auth"])
def me(user: CurrentUser):
    return user_out(user)


@app.post("/auth/change-password", tags=["Auth"])
def change_password(data: ChangePasswordIn, user: CurrentUser, db: DB):
    if not verify_password(data.old_password, user.password_hash):
        raise HTTPException(st.HTTP_400_BAD_REQUEST, "Old password galat hai")
    if data.old_password == data.new_password:
        raise HTTPException(st.HTTP_400_BAD_REQUEST, "New password purane se alag hona chahiye")
    user.password_hash = hash_password(data.new_password)
    db.add(user)
    db.commit()
    return {"detail": "Password change ho gaya"}

# ───────────────────────── Admin ─────────────────────────
@app.post("/admin/students", status_code=201, tags=["Admin"])
def admin_create_student(data: StudentCreate, _: AdminOnly, db: DB):
    try:
        user = create_user(db, email=data.email, password=data.password, full_name=data.full_name,
                           phone=data.phone, role=Role.student)
        s = Student(user_id=user.id, roll_no=data.roll_no, department=data.department, semester=data.semester,
                    section=data.section, dob=data.dob, gender=data.gender, address=data.address,
                    guardian_name=data.guardian_name, guardian_phone=data.guardian_phone)
        db.add(s)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(st.HTTP_409_CONFLICT, "Roll number already exists")
    return student_out(s)


@app.post("/admin/teachers", status_code=201, tags=["Admin"])
def admin_create_teacher(data: TeacherCreate, _: AdminOnly, db: DB):
    try:
        user = create_user(db, email=data.email, password=data.password, full_name=data.full_name,
                           phone=data.phone, role=Role.teacher)
        t = Teacher(user_id=user.id, employee_id=data.employee_id, department=data.department,
                    designation=data.designation, qualification=data.qualification)
        db.add(t)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(st.HTTP_409_CONFLICT, "Employee ID already exists")
    return teacher_out(t)


@app.get("/admin/teachers", tags=["Admin"])
def admin_list_teachers(_: AdminOnly, db: DB, skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    rows = db.scalars(select(Teacher).order_by(Teacher.employee_id).offset(skip).limit(limit)).all()
    return [teacher_out(t) for t in rows]


@app.post("/admin/subjects", status_code=201, tags=["Admin"])
def admin_create_subject(data: SubjectCreate, _: AdminOnly, db: DB):
    if data.teacher_id and not db.get(Teacher, data.teacher_id):
        raise HTTPException(st.HTTP_404_NOT_FOUND, "Teacher not found")
    sub = Subject(**data.model_dump())
    db.add(sub)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(st.HTTP_409_CONFLICT, "Subject code already exists")
    return {"id": sub.id, "code": sub.code, "name": sub.name, "department": sub.department,
            "semester": sub.semester, "teacher_id": sub.teacher_id}


@app.patch("/admin/users/{user_id}/status", tags=["Admin"])
def admin_set_status(user_id: int, data: StatusIn, admin: AdminOnly, db: DB):
    if user_id == admin.id:
        raise HTTPException(st.HTTP_400_BAD_REQUEST, "Apna account disable nahi kar sakte")
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(st.HTTP_404_NOT_FOUND, "User not found")
    u.is_active = data.is_active
    db.commit()
    return {"id": u.id, "is_active": u.is_active}


@app.get("/admin/dashboard", tags=["Admin"])
def admin_dashboard(_: AdminOnly, db: DB):
    count = lambda m: db.scalar(select(func.count()).select_from(m))
    return {"students": count(Student), "teachers": count(Teacher), "subjects": count(Subject),
            "attendance_records": count(Attendance), "score_records": count(Score),
            "face_enrolled_students": db.scalar(select(func.count()).select_from(Student)
                                                .where(Student.face_encoding.is_not(None)))}

# ───────────────────────── Subjects / Teachers ─────────────────────────
@app.get("/subjects", tags=["Subjects"])
def list_subjects(user: CurrentUser, db: DB):
    stmt = select(Subject).order_by(Subject.department, Subject.semester, Subject.code)
    if user.role == Role.teacher:
        t = db.scalar(select(Teacher).where(Teacher.user_id == user.id))
        stmt = stmt.where(Subject.teacher_id == (t.id if t else -1))
    elif user.role == Role.student:
        s = my_student(db, user)
        stmt = stmt.where(Subject.department == s.department, Subject.semester == s.semester)
    return [{"id": x.id, "code": x.code, "name": x.name, "department": x.department,
             "semester": x.semester, "teacher_id": x.teacher_id} for x in db.scalars(stmt)]


@app.get("/teachers/me", tags=["Teachers"])
def teacher_me(user: Annotated[User, Depends(require_roles(Role.teacher))], db: DB):
    t = db.scalar(select(Teacher).where(Teacher.user_id == user.id))
    if not t:
        raise HTTPException(st.HTTP_404_NOT_FOUND, "Teacher profile nahi mila")
    return teacher_out(t)

# ───────────────────────── Students (self: /me routes pehle) ─────────────────────────
@app.get("/students/me", tags=["Student (self)"])
def student_me(user: StudentOnly, db: DB):
    return student_out(my_student(db, user))


@app.get("/students/me/performance", tags=["Student (self)"])
def student_me_performance(user: StudentOnly, db: DB):
    return performance_for(db, my_student(db, user))


@app.get("/students/me/result", tags=["Student (self)"])
def student_me_result(user: StudentOnly, db: DB, semester: Optional[int] = Query(None, ge=1, le=8)):
    return result_for(db, my_student(db, user), semester)

# ───────────────────────── Students (staff / by id) ─────────────────────────
@app.get("/students", tags=["Students"])
def list_students(_: StaffOnly, db: DB, department: Optional[str] = Query(None, max_length=80),
                  semester: Optional[int] = Query(None, ge=1, le=8), q: Optional[str] = Query(None, max_length=50),
                  skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    stmt = select(Student).join(User, Student.user_id == User.id)
    if department:
        stmt = stmt.where(Student.department == department.upper())
    if semester:
        stmt = stmt.where(Student.semester == semester)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(User.full_name.ilike(like), Student.roll_no.ilike(like)))
    return [student_out(s) for s in db.scalars(stmt.order_by(Student.roll_no).offset(skip).limit(limit))]


@app.get("/students/{student_id}", tags=["Students"])
def get_student(student_id: int, user: CurrentUser, db: DB):
    return student_out(student_for(db, user, student_id))


@app.patch("/students/{student_id}", tags=["Students"])
def update_student(student_id: int, body: StudentUpdate, _: AdminOnly, db: DB):
    s = student_for(db, _, student_id)
    data = body.model_dump(exclude_unset=True)
    for k in ("full_name", "phone"):
        if k in data:
            setattr(s.user, k, data.pop(k))
    for k, v in data.items():
        setattr(s, k, v)
    db.commit()
    return student_out(s)


@app.get("/students/{student_id}/performance", tags=["Students"])
def student_performance(student_id: int, user: CurrentUser, db: DB):
    return performance_for(db, student_for(db, user, student_id))


@app.get("/students/{student_id}/result", tags=["Students"])
def student_result(student_id: int, user: CurrentUser, db: DB, semester: Optional[int] = Query(None, ge=1, le=8)):
    return result_for(db, student_for(db, user, student_id), semester)

# ───────────────────────── Face enrollment ─────────────────────────
@app.post("/students/{student_id}/face", tags=["Face"])
def enroll_face(student_id: int, user: StaffOnly, db: DB, image: UploadFile = File(...)):
    """Student ki clear photo (sirf ek chehra) upload karo: face encoding save hoti hai."""
    need_face_lib()
    s = student_for(db, user, student_id)
    img = load_image(image)
    encs = face_recognition.face_encodings(np.array(img))
    if len(encs) != 1:
        raise HTTPException(st.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Photo me exactly 1 chehra hona chahiye (mile: {len(encs)})")
    path = UPLOAD_DIR / "faces" / f"student_{s.id}.jpg"
    img.save(path, "JPEG", quality=90)
    s.face_encoding = [float(x) for x in encs[0]]
    s.photo_path = str(path)
    db.commit()
    return {"detail": "Face enrolled", "student_id": s.id}


@app.get("/students/{student_id}/photo", tags=["Face"])
def student_photo(student_id: int, user: CurrentUser, db: DB):
    s = student_for(db, user, student_id)
    if not s.photo_path or not Path(s.photo_path).exists():
        raise HTTPException(st.HTTP_404_NOT_FOUND, "Photo upload nahi hui")
    return FileResponse(s.photo_path, media_type="image/jpeg")

# ───────────────────────── Attendance ─────────────────────────
@app.post("/attendance/face", tags=["Attendance"])
def face_attendance(user: StaffOnly, db: DB, subject_id: int = Form(..., ge=1),
                    attendance_date: Optional[date] = Form(None), mark_absent_others: bool = Form(False),
                    image: UploadFile = File(...)):
    """Class ki photo upload karo: pehchane gaye students 'present' ho jaate hain."""
    need_face_lib()
    sub = subject_for(db, user, subject_id)
    d = attendance_date or date.today()
    if d > date.today():
        raise HTTPException(st.HTTP_400_BAD_REQUEST, "Future date allowed nahi hai")
    students = db.scalars(select(Student).where(Student.department == sub.department,
                                                Student.semester == sub.semester)).all()
    enrolled = [s for s in students if s.face_encoding]
    if not enrolled:
        raise HTTPException(st.HTTP_400_BAD_REQUEST, "Is subject ke kisi student ka face enrolled nahi hai")
    arr = np.array(load_image(image))
    locs = face_recognition.face_locations(arr)
    if not locs:
        raise HTTPException(st.HTTP_422_UNPROCESSABLE_ENTITY, "Photo me koi chehra detect nahi hua")
    known = np.array([s.face_encoding for s in enrolled])
    matched: dict[int, float] = {}
    unknown = 0
    for enc in face_recognition.face_encodings(arr, locs):
        dist = face_recognition.face_distance(known, enc)
        i = int(np.argmin(dist))
        if dist[i] <= FACE_TOLERANCE:
            sid = enrolled[i].id
            matched[sid] = min(matched.get(sid, 1.0), float(dist[i]))
        else:
            unknown += 1
    existing = existing_attendance(db, sub.id, d)
    absent = 0
    for s in students:
        if s.id in matched:
            upsert_attendance(db, existing, s.id, sub.id, d, AttStatus.present, AttMethod.face, user.id,
                              round(1 - matched[s.id], 3))
        elif mark_absent_others and s.id not in existing:
            upsert_attendance(db, existing, s.id, sub.id, d, AttStatus.absent, AttMethod.face, user.id)
            absent += 1
    db.commit()
    by_id = {s.id: s for s in students}
    return {"date": d, "subject": sub.code, "faces_detected": len(locs), "unrecognized_faces": unknown,
            "marked_absent": absent,
            "present": [{"student_id": i, "roll_no": by_id[i].roll_no, "name": by_id[i].user.full_name,
                         "confidence": round(1 - dist_, 3)} for i, dist_ in matched.items()]}


@app.post("/attendance/manual", tags=["Attendance"])
def manual_attendance(data: ManualAttendanceIn, user: StaffOnly, db: DB):
    sub = subject_for(db, user, data.subject_id)
    ids = [r.student_id for r in data.records]
    studs = {s.id: s for s in db.scalars(select(Student).where(Student.id.in_(ids)))}
    bad = [i for i in ids if i not in studs or studs[i].department != sub.department
           or studs[i].semester != sub.semester]
    if bad:
        raise HTTPException(st.HTTP_400_BAD_REQUEST, f"Ye students is subject ke nahi hain: {bad}")
    existing = existing_attendance(db, sub.id, data.attendance_date)
    for r in data.records:
        upsert_attendance(db, existing, r.student_id, sub.id, data.attendance_date, r.status,
                          AttMethod.manual, user.id)
    db.commit()
    return {"detail": "Attendance saved", "records": len(data.records)}


@app.get("/subjects/{subject_id}/attendance", tags=["Attendance"])
def subject_attendance(subject_id: int, user: StaffOnly, db: DB, on_date: Optional[date] = None):
    sub = subject_for(db, user, subject_id)
    d = on_date or date.today()
    students = db.scalars(select(Student).where(Student.department == sub.department,
                                                Student.semester == sub.semester).order_by(Student.roll_no)).all()
    existing = existing_attendance(db, sub.id, d)
    return {"date": d, "subject": sub.code, "students": [
        {"student_id": s.id, "roll_no": s.roll_no, "name": s.user.full_name,
         "status": existing[s.id].status if s.id in existing else None,
         "method": existing[s.id].method if s.id in existing else None} for s in students]}


@app.get("/students/{student_id}/attendance", tags=["Attendance"])
def student_attendance(student_id: int, user: CurrentUser, db: DB, subject_id: Optional[int] = Query(None, ge=1),
                       start: Optional[date] = None, end: Optional[date] = None,
                       skip: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)):
    s = student_for(db, user, student_id)
    if start and end and start > end:
        raise HTTPException(st.HTTP_400_BAD_REQUEST, "start, end se pehle hona chahiye")
    stmt = (select(Attendance, Subject.code).join(Subject, Subject.id == Attendance.subject_id)
            .where(Attendance.student_id == s.id))
    if subject_id:
        stmt = stmt.where(Attendance.subject_id == subject_id)
    if start:
        stmt = stmt.where(Attendance.date >= start)
    if end:
        stmt = stmt.where(Attendance.date <= end)
    rows = db.execute(stmt.order_by(Attendance.date.desc()).offset(skip).limit(limit)).all()
    return [{"date": a.date, "subject": code, "status": a.status, "method": a.method,
             "confidence": a.confidence} for a, code in rows]

# ───────────────────────── Scores ─────────────────────────
@app.post("/scores", status_code=201, tags=["Scores"])
def add_score(data: ScoreIn, user: StaffOnly, db: DB):
    sub = subject_for(db, user, data.subject_id)
    s = db.get(Student, data.student_id)
    if not s or s.department != sub.department or s.semester != sub.semester:
        raise HTTPException(st.HTTP_400_BAD_REQUEST, "Student is subject ka nahi hai")
    if db.scalar(select(Score.id).where(Score.student_id == s.id, Score.subject_id == sub.id,
                                        Score.exam_name == data.exam_name)):
        raise HTTPException(st.HTTP_409_CONFLICT, "Is exam ka score pehle se hai (PUT se update karo)")
    sc = Score(**data.model_dump(), recorded_by=user.id)
    db.add(sc)
    db.commit()
    return {"id": sc.id, "percentage": pct(sc.marks_obtained, sc.max_marks), "grade": grade(pct(sc.marks_obtained, sc.max_marks))}


@app.put("/scores/{score_id}", tags=["Scores"])
def update_score(score_id: int, body: ScoreUpdate, user: StaffOnly, db: DB):
    sc = db.get(Score, score_id)
    if not sc:
        raise HTTPException(st.HTTP_404_NOT_FOUND, "Score not found")
    subject_for(db, user, sc.subject_id)
    data = body.model_dump(exclude_unset=True)
    marks = data.get("marks_obtained", sc.marks_obtained)
    mx = data.get("max_marks", sc.max_marks)
    if marks > mx:
        raise HTTPException(st.HTTP_422_UNPROCESSABLE_ENTITY, "marks_obtained, max_marks se zyada nahi ho sakte")
    for k, v in data.items():
        setattr(sc, k, v)
    db.commit()
    return {"id": sc.id, "marks_obtained": sc.marks_obtained, "max_marks": sc.max_marks,
            "percentage": pct(sc.marks_obtained, sc.max_marks)}


@app.get("/students/{student_id}/scores", tags=["Scores"])
def student_scores(student_id: int, user: CurrentUser, db: DB, subject_id: Optional[int] = Query(None, ge=1)):
    s = student_for(db, user, student_id)
    stmt = (select(Score, Subject.code, Subject.name).join(Subject, Subject.id == Score.subject_id)
            .where(Score.student_id == s.id))
    if subject_id:
        stmt = stmt.where(Score.subject_id == subject_id)
    return [{"id": sc.id, "subject": code, "subject_name": name, "exam_type": sc.exam_type,
             "exam_name": sc.exam_name, "marks_obtained": sc.marks_obtained, "max_marks": sc.max_marks,
             "percentage": pct(sc.marks_obtained, sc.max_marks), "exam_date": sc.exam_date,
             "remarks": sc.remarks} for sc, code, name in db.execute(stmt.order_by(Score.exam_date.desc()))]
