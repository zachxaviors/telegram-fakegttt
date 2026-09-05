"""
================================================================================
BOT TELEGRAM ĐIỀU PHỐI HỆ THỐNG (ORCHESTRATOR BOT)
================================================================================
Mục đích:
    - Đóng vai trò "bộ não điều phối" trung tâm, kết nối người dùng Telegram
      với một Telegram Mini App (WebApp) để họ upload ảnh + nhập text chỉnh sửa.
    - Sau khi nhận dữ liệu từ WebApp, Bot sẽ gọi sang một pipeline AI (Vision
      model để bóc tách tọa độ + Image Edit model để sửa ảnh) thông qua cổng
      OpenAI-Compatible API (CKEY).
    - Toàn bộ luồng xử lý được viết bất đồng bộ (async/await) để không làm
      nghẽn Bot khi có nhiều người dùng thao tác cùng lúc.

Kiến trúc:
    - python-telegram-bot >= 20 (Application, ContextTypes, async handlers)
    - httpx.AsyncClient để gọi API Backend không đồng bộ (non-blocking)
    - logging để ghi nhận toàn bộ tiến trình / lỗi phát sinh ra console

Lưu ý quan trọng:
    - Các hàm gọi model AI thật (Qwen Vision, FLUX Image Edit) hiện đang ở
      dạng STUB/MOCK để bạn tự đấu nối sau. Vị trí cần chỉnh sửa được đánh
      dấu rõ ràng bằng comment "TODO(CKEY)".
================================================================================
"""

import asyncio
import base64
import io
import json
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple

import httpx
from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
    WebAppInfo,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)


# ==============================================================================
# 1. CẤU HÌNH BIẾN MÔI TRƯỜNG (CONFIGURATION)
# ==============================================================================
# --- Tải biến môi trường từ file .env ---
load_dotenv()

# --- Token của Bot Telegram, lấy từ @BotFather ---
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

# --- Link trang WebApp (Mini App) tạm thời, PHẢI là HTTPS ---
WEB_APP_URL: str = os.getenv("WEB_APP_URL", "https://zachxaviors.github.io/telegram-fakegttt/")

# --- URL FastAPI Backend ---
BACKEND_API_URL: str = os.getenv("BACKEND_API_URL", "http://localhost:8080")

# --- Internal API Key for backend auth ---
INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "bot-internal-secret-key-2026")

# --- ID Telegram của (các) người dùng được phép dùng lệnh /key ---
AUTHORIZED_KEY_ADMIN_IDS: list[int] = [int(x) for x in os.getenv("AUTHORIZED_KEY_ADMIN_IDS", "8329365661").split(",")]

# --- Số ngày hiệu lực mặc định cho mỗi Key được tạo qua lệnh /key ---
KEY_VALID_DAYS: int = int(os.getenv("KEY_VALID_DAYS", "30"))

# --- Cấu hình GitHub để Bot tự động đồng bộ Key vào file keys.json ---
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO_OWNER: str = os.getenv("GITHUB_REPO_OWNER", "zachxaviors")
GITHUB_REPO_NAME: str = os.getenv("GITHUB_REPO_NAME", "telegram-fakegttt")
GITHUB_REPO_BRANCH: str = os.getenv("GITHUB_REPO_BRANCH", "main")
GITHUB_KEYS_FILE_PATH: str = os.getenv("GITHUB_KEYS_FILE_PATH", "keys.json")


# ==============================================================================
# 2. CẤU HÌNH LOGGING
# ==============================================================================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# Giảm bớt log rác từ thư viện httpx (chỉ hiện WARNING trở lên)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("OrchestratorBot")


# ==============================================================================
# 3. HÀM GỌI API QWEN2.5-VL / MIMO-V2.5 (VISION MODEL)
# ==============================================================================
async def call_backend_inpaint(image_url: str, new_text: str, progress_callback=None) -> Optional[io.BytesIO]:
    if progress_callback:
        await progress_callback("📥 Đang tải ảnh gốc...")
    
    async with httpx.AsyncClient(timeout=30.0) as img_client:
        img_resp = await img_client.get(image_url)
        img_resp.raise_for_status()
        image_bytes = img_resp.content
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    if progress_callback:
        await progress_callback("🔍 Đang phân tích tọa độ và font chữ...")

    async with httpx.AsyncClient(timeout=180.0, headers={"Authorization": f"Bearer {INTERNAL_API_KEY}"}) as client:
        ocr_resp = await client.post(
            f"{BACKEND_API_URL}/ocr",
            json={"image_base64": image_base64},
        )
        ocr_resp.raise_for_status()
        ocr_data = ocr_resp.json()
        coordinates = ocr_data.get("coordinates", {})

        if not coordinates:
            logger.error("Backend /ocr returned no coordinates")
            return None

        if progress_callback:
            await progress_callback("✏️ Đang xóa chữ cũ và tái tạo ký tự mới...\n⏳ Vui lòng chờ 10-30 giây...")

        inpaint_resp = await client.post(
            f"{BACKEND_API_URL}/inpaint",
            json={
                "image_base64": image_base64,
                "coordinates": coordinates,
                "prompt": new_text,
            },
        )
        inpaint_resp.raise_for_status()

        render_method = inpaint_resp.headers.get("X-Render-Method", "unknown")
        processing_ms = inpaint_resp.headers.get("X-Processing-Time-Ms", "?")
        
        if progress_callback:
            method_labels = {
                "local-font": "font matching cục bộ",
                "glyph-composite": "tái tạo ký tự từ ảnh gốc",
                "api-inpaint": "AI inpainting",
            }
            method_label = method_labels.get(render_method, render_method)
            await progress_callback(f"🎨 Đang hoàn thiện ảnh ({method_label}, {processing_ms}ms)...")

        buf = io.BytesIO(inpaint_resp.content)
        buf.seek(0)
        return buf


# ==============================================================================
# 4. HÀM ĐỒNG BỘ KEY VỚI GITHUB (keys.json trên repo chứa WebApp)
# ==============================================================================
def _build_github_headers() -> dict:
    """Tạo headers chuẩn dùng chung cho mọi lời gọi GitHub Contents API."""
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _build_github_keys_url() -> str:
    """Tạo URL endpoint tới file keys.json trên GitHub Contents API."""
    return (
        f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}"
        f"/contents/{GITHUB_KEYS_FILE_PATH}"
    )


async def fetch_keys_from_github() -> Optional[Tuple[dict, str]]:
    """
    Tải nội dung hiện tại của file `keys.json` trên GitHub.

    Returns:
        Optional[Tuple[dict, str]]: Tuple (keys_data, current_sha) nếu thành
        công, trong đó keys_data có dạng {"keys": [{"key":..., "expires":...}]}
        và current_sha là mã băm bắt buộc phải gửi kèm khi ghi đè file.
        Trả về None nếu có lỗi (mất mạng, chưa cấu hình token, v.v.).
    """
    if not GITHUB_TOKEN or GITHUB_TOKEN == "YOUR_GITHUB_PERSONAL_ACCESS_TOKEN_HERE":
        logger.warning("Chưa cấu hình GITHUB_TOKEN, không thể thao tác với keys.json.")
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            get_response = await client.get(
                _build_github_keys_url(),
                headers=_build_github_headers(),
                params={"ref": GITHUB_REPO_BRANCH},
            )
            get_response.raise_for_status()
            file_data = get_response.json()

            current_sha = file_data["sha"]
            current_content_json = base64.b64decode(file_data["content"]).decode("utf-8")
            keys_data: dict = json.loads(current_content_json)
            keys_data.setdefault("keys", [])
            return keys_data, current_sha

    except httpx.HTTPStatusError as http_err:
        logger.error(
            "Lỗi HTTP khi tải keys.json từ GitHub: %s | Response: %s",
            http_err, http_err.response.text,
        )
        return None
    except Exception as e:
        logger.exception("Lỗi không xác định khi tải keys.json từ GitHub: %s", e)
        return None


async def save_keys_to_github(keys_data: dict, current_sha: str, commit_message: str) -> bool:
    """
    Ghi đè nội dung mới của `keys_data` lên file `keys.json` trên GitHub.

    Args:
        keys_data (dict): Toàn bộ nội dung JSON mới (dạng {"keys": [...]}).
        current_sha (str): sha của file hiện tại (lấy từ fetch_keys_from_github),
            bắt buộc phải đúng để tránh xung đột khi ghi đè.
        commit_message (str): Nội dung commit message hiển thị trên GitHub.

    Returns:
        bool: True nếu ghi thành công, False nếu có lỗi.
    """
    try:
        updated_content_json = json.dumps(keys_data, ensure_ascii=False, indent=2)
        updated_content_b64 = base64.b64encode(
            updated_content_json.encode("utf-8")
        ).decode("utf-8")

        put_payload = {
            "message": commit_message,
            "content": updated_content_b64,
            "sha": current_sha,
            "branch": GITHUB_REPO_BRANCH,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            put_response = await client.put(
                _build_github_keys_url(), headers=_build_github_headers(), json=put_payload
            )
            put_response.raise_for_status()

        return True

    except httpx.HTTPStatusError as http_err:
        logger.error(
            "Lỗi HTTP khi ghi keys.json lên GitHub: %s | Response: %s",
            http_err, http_err.response.text,
        )
        return False
    except Exception as e:
        logger.exception("Lỗi không xác định khi ghi keys.json lên GitHub: %s", e)
        return False


async def add_key_to_github(new_key: str, expires: str) -> bool:
    """
    Thêm 1 Key mới vào file `keys.json` trên repo GitHub chứa WebApp.
    WebApp (index.html) sẽ fetch file này để kiểm tra Key hợp lệ, nên Key
    vừa sinh ra từ lệnh /key sẽ có hiệu lực ngay trên Web mà không cần sửa tay.

    Args:
        new_key (str): Chuỗi Key vừa sinh (đã format, ví dụ "7K9X-Q2FZ-4B").
        expires (str): Ngày hết hạn dạng "YYYY-MM-DD".

    Returns:
        bool: True nếu đồng bộ thành công, False nếu có lỗi.
    """
    fetched = await fetch_keys_from_github()
    if fetched is None:
        return False

    keys_data, current_sha = fetched
    keys_data["keys"].append({"key": new_key, "expires": expires})

    success = await save_keys_to_github(
        keys_data, current_sha, f"Bot: thêm Key thuê bao mới ({new_key})"
    )
    if success:
        logger.info("Đã đồng bộ Key '%s' lên GitHub thành công.", new_key)
    return success


async def delete_key_from_github(target_key: str) -> bool:
    """
    Xoá 1 Key khỏi file `keys.json` trên GitHub theo đúng chuỗi Key.

    Args:
        target_key (str): Chuỗi Key cần xoá (so khớp không phân biệt HOA/thường).

    Returns:
        bool: True nếu xoá + ghi lại thành công, False nếu Key không tồn tại
        hoặc có lỗi khi thao tác với GitHub.
    """
    fetched = await fetch_keys_from_github()
    if fetched is None:
        return False

    keys_data, current_sha = fetched
    normalized_target = target_key.strip().upper()

    original_count = len(keys_data["keys"])
    keys_data["keys"] = [
        item for item in keys_data["keys"]
        if (item.get("key") or "").upper() != normalized_target
    ]

    if len(keys_data["keys"]) == original_count:
        # Không tìm thấy Key nào khớp để xoá
        logger.warning("Không tìm thấy Key '%s' để xoá trên keys.json.", target_key)
        return False

    success = await save_keys_to_github(
        keys_data, current_sha, f"Bot: xoá Key thuê bao ({target_key})"
    )
    if success:
        logger.info("Đã xoá Key '%s' khỏi GitHub thành công.", target_key)
    return success


# ==============================================================================
# 5. HÀM SINH KEY THUÊ BAO NGẮN GỌN
# ==============================================================================
def generate_license_key(length: int = 10) -> str:
    """
    Sinh một chuỗi Key ngắn gọn, dễ đọc, dễ gõ tay (chỉ dùng chữ HOA + số,
    loại bỏ các ký tự dễ gây nhầm lẫn như 0/O, 1/I).

    Returns:
        str: Chuỗi Key, ví dụ "7K9XQ2FZ4B".
    """
    # Bảng ký tự an toàn (đã loại 0, O, 1, I để tránh nhầm lẫn khi đọc/gõ)
    safe_chars = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "".join(random.choices(safe_chars, k=length))


# ==============================================================================
# 6. HANDLER: LỆNH /key - CẤP KEY THUÊ BAO (CHỈ ADMIN ĐƯỢC PHÉP)
# ==============================================================================
async def key_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Xử lý lệnh /key.
    Sinh ra một mã Key thuê bao ngắn gọn kèm ngày hết hạn để admin cấp cho
    khách thuê. CHỈ những user có ID nằm trong AUTHORIZED_KEY_ADMIN_IDS mới
    được phép sử dụng lệnh này, mọi user khác sẽ bị từ chối.

    Cách dùng:
        /key           -> tạo Key với số ngày hiệu lực mặc định (KEY_VALID_DAYS)
        /key 15        -> tạo Key có hiệu lực 15 ngày kể từ hôm nay
        /key 90        -> tạo Key có hiệu lực 90 ngày kể từ hôm nay
    """
    user = update.effective_user
    logger.info("Người dùng %s (ID: %s) đã gọi lệnh /key", user.full_name, user.id)

    # --- Kiểm tra quyền hạn: chỉ admin trong danh sách mới được dùng lệnh ---
    if user.id not in AUTHORIZED_KEY_ADMIN_IDS:
        logger.warning("Từ chối lệnh /key: user ID %s không có quyền.", user.id)
        await update.message.reply_text(
            "⛔ Bạn không có quyền sử dụng lệnh này."
        )
        return

    try:
        # --- Xác định số ngày hiệu lực: lấy từ tham số /key <số_ngày>, ---
        # --- nếu không nhập hoặc nhập sai định dạng thì dùng mặc định. ---
        valid_days = KEY_VALID_DAYS
        if context.args:
            try:
                requested_days = int(context.args[0])
                if requested_days <= 0:
                    raise ValueError("Số ngày phải lớn hơn 0")
                valid_days = requested_days
            except ValueError:
                await update.message.reply_text(
                    "⚠️ Số ngày không hợp lệ. Cú pháp: `/key <số_ngày>`, "
                    f"ví dụ `/key 30`. Đang dùng mặc định {KEY_VALID_DAYS} ngày.",
                    parse_mode=ParseMode.MARKDOWN,
                )

        # --- Sinh Key mới + tính ngày hết hạn theo số ngày đã xác định ---
        new_key = generate_license_key()
        expiry_date = (datetime.now() + timedelta(days=valid_days)).strftime("%Y-%m-%d")

        # --- Định dạng lại để hiển thị dễ đọc, ví dụ: 7K9X-Q2FZ-4B ---
        formatted_key = "-".join(
            [new_key[i:i + 4] for i in range(0, len(new_key), 4)]
        )

        # --- Báo cho khách biết đang đồng bộ Key lên Web (có thể mất vài giây) ---
        syncing_message = await update.message.reply_text(
            "⏳ Đang tạo Key và đồng bộ lên WebApp..."
        )

        # --- Tự động đẩy Key mới lên file keys.json trên GitHub ---
        sync_success = await add_key_to_github(formatted_key, expiry_date)

        if sync_success:
            sync_status_line = "✅ Đã đồng bộ lên WebApp, khách có thể dùng ngay."
        else:
            sync_status_line = (
                "⚠️ Đồng bộ lên WebApp thất bại. Kiểm tra log Bot hoặc thêm "
                "Key thủ công vào `keys.json` trên GitHub."
            )

        reply_text = (
            "🔑 *Key thuê bao mới đã được tạo:*\n\n"
            f"`{formatted_key}`\n\n"
            f"⏳ Thời hạn: *{valid_days} ngày*\n"
            f"📅 Hiệu lực đến: *{expiry_date}*\n\n"
            f"{sync_status_line}\n\n"
            "👉 Gửi Key này cho khách thuê."
        )

        try:
            await syncing_message.delete()
        except Exception:
            pass

        await update.message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN)
        logger.info(
            "Đã cấp Key mới: %s | Thời hạn: %d ngày | Hết hạn: %s | Đồng bộ GitHub: %s",
            formatted_key, valid_days, expiry_date, sync_success,
        )

    except Exception as e:
        logger.exception("Lỗi khi xử lý lệnh /key: %s", e)
        await update.message.reply_text(
            "⚠️ Đã có lỗi xảy ra khi tạo Key. Vui lòng thử lại sau."
        )


def _build_listkey_message(keys_data: dict) -> Tuple[str, Optional[InlineKeyboardMarkup]]:
    """
    Dựng sẵn nội dung text + bàn phím inline (nút xoá) cho danh sách Key,
    dùng chung cho cả lệnh /listkey và khi refresh lại sau khi xoá 1 Key.

    Returns:
        Tuple[str, Optional[InlineKeyboardMarkup]]: (nội dung tin nhắn, bàn
        phím inline kèm nút xoá cho từng Key; None nếu danh sách rỗng).
    """
    keys_list = keys_data.get("keys", [])

    if not keys_list:
        return "📭 Hiện chưa có Key thuê bao nào.", None

    today = datetime.now().date()
    lines = ["🗂 *Danh sách Key thuê bao:*\n"]
    buttons = []

    for index, item in enumerate(keys_list, start=1):
        key_value = item.get("key", "???")
        expires = item.get("expires", "???")

        # Đánh dấu trạng thái còn hạn / đã hết hạn để dễ theo dõi
        try:
            expiry_date = datetime.strptime(expires, "%Y-%m-%d").date()
            status_icon = "✅" if expiry_date >= today else "❌"
        except ValueError:
            status_icon = "❔"

        lines.append(f"{index}. {status_icon} `{key_value}` — hết hạn {expires}")

        # Mỗi Key có 1 nút xoá riêng, callback_data mang theo chuỗi Key
        # để nhận diện chính xác Key nào cần xoá khi người dùng bấm.
        buttons.append(
            [InlineKeyboardButton(f"🗑 Xoá {key_value}", callback_data=f"delkey:{key_value}")]
        )

    message_text = "\n".join(lines)
    return message_text, InlineKeyboardMarkup(buttons)


# ==============================================================================
# 7. HANDLER: LỆNH /listkey - XEM DANH SÁCH & XOÁ KEY (CHỈ ADMIN ĐƯỢC PHÉP)
# ==============================================================================
async def listkey_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Xử lý lệnh /listkey.
    Hiển thị toàn bộ danh sách Key thuê bao hiện có trên keys.json (GitHub),
    kèm theo 1 nút "🗑 Xoá" riêng cho từng Key để admin xoá nhanh ngay trong
    Telegram. CHỈ những user có ID nằm trong AUTHORIZED_KEY_ADMIN_IDS mới
    được phép sử dụng lệnh này.
    """
    user = update.effective_user
    logger.info("Người dùng %s (ID: %s) đã gọi lệnh /listkey", user.full_name, user.id)

    if user.id not in AUTHORIZED_KEY_ADMIN_IDS:
        logger.warning("Từ chối lệnh /listkey: user ID %s không có quyền.", user.id)
        await update.message.reply_text("⛔ Bạn không có quyền sử dụng lệnh này.")
        return

    try:
        fetched = await fetch_keys_from_github()
        if fetched is None:
            await update.message.reply_text(
                "❌ Không tải được danh sách Key từ GitHub. Vui lòng thử lại sau."
            )
            return

        keys_data, _ = fetched
        message_text, reply_markup = _build_listkey_message(keys_data)

        await update.message.reply_text(
            message_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
        )

    except Exception as e:
        logger.exception("Lỗi khi xử lý lệnh /listkey: %s", e)
        await update.message.reply_text(
            "⚠️ Đã có lỗi xảy ra khi tải danh sách Key. Vui lòng thử lại sau."
        )


# ==============================================================================
# 8. HANDLER: CALLBACK XOÁ KEY (khi bấm nút "🗑 Xoá" trong /listkey)
# ==============================================================================
async def delete_key_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Xử lý sự kiện bấm nút inline "🗑 Xoá <Key>" được sinh ra từ /listkey.
    Xoá Key tương ứng khỏi keys.json trên GitHub, sau đó cập nhật lại ngay
    tin nhắn danh sách Key để phản ánh thay đổi mới nhất (không cần gõ lại
    /listkey lần nữa).
    """
    query = update.callback_query
    user = query.from_user

    # --- Kiểm tra quyền hạn: chỉ admin trong danh sách mới được xoá Key ---
    if user.id not in AUTHORIZED_KEY_ADMIN_IDS:
        await query.answer("⛔ Bạn không có quyền thực hiện thao tác này.", show_alert=True)
        return

    # callback_data có dạng "delkey:<KEY_VALUE>"
    target_key = query.data.split(":", 1)[1] if ":" in query.data else ""

    if not target_key:
        await query.answer("❌ Dữ liệu Key không hợp lệ.", show_alert=True)
        return

    # --- Báo cho admin biết đang xử lý (hiện toast loading nhỏ) ---
    await query.answer("⏳ Đang xoá Key...")

    try:
        success = await delete_key_from_github(target_key)

        if not success:
            await query.answer(
                f"❌ Xoá Key '{target_key}' thất bại hoặc Key không tồn tại.",
                show_alert=True,
            )
            return

        logger.info("Admin %s (ID: %s) đã xoá Key '%s'", user.full_name, user.id, target_key)

        # --- Tải lại danh sách Key mới nhất và cập nhật lại tin nhắn cũ ---
        fetched = await fetch_keys_from_github()
        if fetched is not None:
            keys_data, _ = fetched
            message_text, reply_markup = _build_listkey_message(keys_data)
            await query.edit_message_text(
                text=f"✅ Đã xoá Key `{target_key}`.\n\n{message_text}",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup,
            )
        else:
            await query.edit_message_text(
                text=f"✅ Đã xoá Key `{target_key}`, nhưng không tải lại được danh sách mới nhất.",
                parse_mode=ParseMode.MARKDOWN,
            )

    except Exception as e:
        logger.exception("Lỗi khi xử lý callback xoá Key: %s", e)
        await query.answer("⚠️ Đã có lỗi xảy ra khi xoá Key.", show_alert=True)


# ==============================================================================
# 9. HANDLER: LỆNH /start - GỬI NÚT MỞ WEBAPP
# ==============================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Xử lý lệnh /start.
    Gửi tin nhắn chào mừng kèm nút bấm mở WebApp (Telegram Mini App) trượt
    từ dưới lên, cho phép người dùng upload ảnh + nhập text ngay trong Telegram.
    """
    user = update.effective_user
    logger.info("Người dùng %s (ID: %s) đã gọi lệnh /start", user.full_name, user.id)

    try:
        # --- Gắn tham số "?v=<timestamp>" vào cuối URL để buộc Telegram luôn
        # --- tải bản index.html MỚI NHẤT từ GitHub Pages, tránh trường hợp
        # --- Telegram cache lại bản HTML/JS cũ (khiến Key mới cấp không
        # --- nhận được vì WebApp vẫn chạy code cũ đã cache trước đó). ---
        cache_busted_url = f"{WEB_APP_URL}?v={int(datetime.now().timestamp())}"

        # --- Tạo nút bấm đặc biệt chứa WebAppInfo để mở Mini App ---
        web_app_button = KeyboardButton(
            text="📷 Mở công cụ chỉnh sửa ảnh",
            web_app=WebAppInfo(url=cache_busted_url),
        )

        # --- Bàn phím tùy chỉnh (Reply Keyboard) chỉ chứa 1 nút WebApp ---
        reply_markup = ReplyKeyboardMarkup(
            keyboard=[[web_app_button]],
            resize_keyboard=True,
            one_time_keyboard=False,
        )

        welcome_text = (
            f"👋 Xin chào *{user.first_name}*!\n\n"
            "Chào mừng bạn đến với hệ thống chỉnh sửa ảnh tự động bằng AI.\n\n"
            "👉 Nhấn vào nút bên dưới để mở công cụ, tải ảnh lên và nhập "
            "nội dung chữ mới bạn muốn thay thế trên ảnh."
        )

        await update.message.reply_text(
            text=welcome_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN,
        )

    except Exception as e:
        logger.exception("Lỗi khi xử lý lệnh /start: %s", e)
        await update.message.reply_text(
            "⚠️ Đã có lỗi xảy ra khi khởi tạo Bot. Vui lòng thử lại sau."
        )


# ==============================================================================
# 10. HANDLER: CỔNG TIẾP NHẬN DỮ LIỆU TỪ WEBAPP
# ==============================================================================
async def web_app_data_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    "Cổng tiếp nhận dữ liệu" từ WebApp.
    Khi người dùng thao tác xong trên WebApp (upload ảnh + nhập text) và bấm
    xác nhận, WebApp sẽ gọi `Telegram.WebApp.sendData(...)` gửi một chuỗi JSON
    về Bot thông qua Update có `message.web_app_data`.

    Cấu trúc JSON kỳ vọng nhận được từ WebApp (phía Frontend cần đóng gói đúng):
        {
            "image_url": "<link ảnh công khai do WebApp upload lên ImgBB>",
            "new_text": "<văn bản mới người dùng nhập>"
        }
    """
    chat_id = update.effective_chat.id
    user = update.effective_user

    try:
        # --- Bước 1: Lấy chuỗi JSON thô do WebApp gửi về ---
        raw_data: str = update.message.web_app_data.data
        logger.info(
            "Nhận dữ liệu WebApp từ user %s (ID: %s) | Kích thước chuỗi: %d ký tự",
            user.full_name, user.id, len(raw_data),
        )

        # --- Bước 2: Giải mã (parse) chuỗi JSON để tách 2 thông tin cần thiết ---
        try:
            payload: dict = json.loads(raw_data)
        except json.JSONDecodeError as json_err:
            logger.error("Dữ liệu WebApp không phải JSON hợp lệ: %s", json_err)
            await update.message.reply_text(
                "⚠️ Dữ liệu gửi về từ WebApp không hợp lệ. Vui lòng thử lại."
            )
            return

        image_url: Optional[str] = payload.get("image_url")
        new_text: Optional[str] = payload.get("new_text")

        # --- Bước 3: Kiểm tra tính hợp lệ của dữ liệu trước khi xử lý tiếp ---
        if not image_url or not new_text:
            logger.warning(
                "Dữ liệu WebApp thiếu trường bắt buộc | image_url=%s | new_text=%s",
                bool(image_url), bool(new_text),
            )
            await update.message.reply_text(
                "⚠️ Thiếu dữ liệu ảnh hoặc văn bản. Vui lòng thao tác lại trên WebApp."
            )
            return

        # --- Bước 4: Phản hồi ngay lập tức cho khách biết hệ thống đã nhận ---
        processing_message = await update.message.reply_text(
            "✅ Đã nhận được ảnh và nội dung của bạn!\n"
            "🤖 Hệ thống AI đang xử lý, vui lòng chờ trong giây lát..."
        )

        async def update_progress(status_text: str):
            try:
                await processing_message.edit_text(status_text)
            except Exception:
                pass

        # --- Bước 5: Gửi hiệu ứng "đang tải ảnh" để trải nghiệm mượt hơn ---
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)

        # --- Bước 6: Gọi FastAPI backend (OCR + Inpaint) ---
        result_image_buffer = await call_backend_inpaint(
            image_url=image_url,
            new_text=new_text,
            progress_callback=update_progress,
        )

        # --- Bước 7: Trả kết quả về cho người dùng (in-memory binary, no disk) ---
        if result_image_buffer:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=result_image_buffer,
                caption="🎉 Xử lý hoàn tất! Đây là ảnh kết quả của bạn.",
            )
            logger.info("Đã gửi ảnh kết quả thành công cho user %s", user.id)
        else:
            await update.message.reply_text(
                "❌ Rất tiếc, hệ thống AI xử lý ảnh gặp lỗi. Vui lòng thử lại sau."
            )
            logger.error("Pipeline AI trả về None cho user %s", user.id)

        # --- Dọn dẹp: xóa tin nhắn "đang xử lý" cho gọn khung chat (tùy chọn) ---
        try:
            await processing_message.delete()
        except Exception:
            # Không nghiêm trọng nếu xóa thất bại (VD: tin nhắn đã bị xóa tay)
            pass

    except Exception as e:
        logger.exception("Lỗi không xác định trong web_app_data_handler: %s", e)
        await update.message.reply_text(
            "❌ Đã có lỗi hệ thống xảy ra. Vui lòng thử lại hoặc liên hệ hỗ trợ."
        )


# ==============================================================================
# 11. HANDLER: BẮT LỖI TOÀN CỤC (GLOBAL ERROR HANDLER)
# ==============================================================================
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Bắt và ghi log toàn bộ lỗi phát sinh trong quá trình Bot chạy mà không
    được xử lý ở các handler riêng lẻ, tránh làm sập tiến trình Bot.
    """
    logger.error("Lỗi toàn cục xảy ra khi xử lý update: %s", update, exc_info=context.error)


# ==============================================================================
# 12. HANDLER: NHẬN ẢNH TRỰC TIẾP TỪ TIN NHẮN + DEBUG HTML
# ==============================================================================
async def photo_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user = update.effective_user
    caption = update.message.caption or ""

    if not caption.strip():
        await update.message.reply_text(
            "⚠️ Vui lòng gửi ảnh kèm caption chứa text mới muốn thay thế.\n"
            "Ví dụ: Gửi ảnh CCCD + caption \"NGUYEN VAN A\""
        )
        return

    processing_msg = await update.message.reply_text(
        "📥 Đã nhận ảnh! Đang xử lý...\n⏳ Vui lòng chờ 10-30 giây."
    )

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)

        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()
        image_base64 = base64.b64encode(bytes(image_bytes)).decode("utf-8")

        async def update_progress(status_text: str):
            try:
                await processing_msg.edit_text(status_text)
            except Exception:
                pass

        await update_progress("🔍 Đang phân tích tọa độ và font chữ...")

        async with httpx.AsyncClient(timeout=180.0, headers={"Authorization": f"Bearer {INTERNAL_API_KEY}"}) as client:
            ocr_resp = await client.post(
                f"{BACKEND_API_URL}/ocr",
                json={"image_base64": image_base64},
            )

            if ocr_resp.status_code != 200:
                error_html = generate_debug_html(
                    image_base64=image_base64,
                    coordinates={},
                    error=f"OCR Error: {ocr_resp.status_code} - {ocr_resp.text[:500]}",
                    stage="ocr",
                )
                await processing_msg.edit_text(
                    f"❌ Lỗi OCR (HTTP {ocr_resp.status_code}).\nĐã tạo file debug.",
                )
                debug_buf = io.BytesIO(error_html.encode("utf-8"))
                debug_buf.name = "debug_ocr.html"
                await context.bot.send_document(chat_id=chat_id, document=debug_buf, filename="debug_ocr.html")
                return

            ocr_data = ocr_resp.json()
            coordinates = ocr_data.get("coordinates", {})

            if not coordinates:
                error_html = generate_debug_html(
                    image_base64=image_base64,
                    coordinates={},
                    error="OCR returned empty coordinates",
                    stage="ocr",
                )
                await processing_msg.edit_text("❌ Không tìm thấy tọa độ text trên ảnh.")
                debug_buf = io.BytesIO(error_html.encode("utf-8"))
                debug_buf.name = "debug_ocr.html"
                await context.bot.send_document(chat_id=chat_id, document=debug_buf, filename="debug_ocr.html")
                return

            await update_progress(f"✏️ Tọa độ: {json.dumps(coordinates)}\n⏳ Đang xóa chữ cũ và tái tạo ký tự...")

            inpaint_resp = await client.post(
                f"{BACKEND_API_URL}/inpaint",
                json={
                    "image_base64": image_base64,
                    "coordinates": coordinates,
                    "prompt": caption.strip(),
                },
            )

            if inpaint_resp.status_code != 200:
                error_html = generate_debug_html(
                    image_base64=image_base64,
                    coordinates=coordinates,
                    error=f"Inpaint Error: {inpaint_resp.status_code} - {inpaint_resp.text[:500]}",
                    stage="inpaint",
                )
                await processing_msg.edit_text(f"❌ Lỗi Inpaint (HTTP {inpaint_resp.status_code}).")
                debug_buf = io.BytesIO(error_html.encode("utf-8"))
                debug_buf.name = "debug_inpaint.html"
                await context.bot.send_document(chat_id=chat_id, document=debug_buf, filename="debug_inpaint.html")
                return

            render_method = inpaint_resp.headers.get("X-Render-Method", "unknown")
            processing_ms = inpaint_resp.headers.get("X-Processing-Time-Ms", "?")

            result_buf = io.BytesIO(inpaint_resp.content)
            result_buf.seek(0)

            debug_html = generate_debug_html(
                image_base64=image_base64,
                coordinates=coordinates,
                result_image_base64=base64.b64encode(inpaint_resp.content).decode("utf-8"),
                render_method=render_method,
                processing_ms=processing_ms,
                new_text=caption.strip(),
                stage="success",
            )

        await processing_msg.delete()

        await context.bot.send_photo(
            chat_id=chat_id,
            photo=result_buf,
            caption=f"🎉 Xử lý hoàn tất!\n🔧 Phương pháp: {render_method}\n⏱ Thời gian: {processing_ms}ms",
        )

        debug_buf = io.BytesIO(debug_html.encode("utf-8"))
        debug_buf.name = "debug_result.html"
        await context.bot.send_document(
            chat_id=chat_id,
            document=debug_buf,
            filename="debug_result.html",
            caption="📋 File debug: mở trên trình duyệt để xem tọa độ bounding box + so sánh ảnh gốc/kết quả.",
        )

        logger.info(f"Photo message processed for user {user.id}: method={render_method}, time={processing_ms}ms")

    except Exception as e:
        logger.exception(f"Error processing photo message: {e}")
        await processing_msg.edit_text(f"❌ Lỗi xử lý: {str(e)[:200]}")


def generate_debug_html(
    image_base64: str,
    coordinates: dict,
    error: str = "",
    stage: str = "",
    result_image_base64: str = "",
    render_method: str = "",
    processing_ms: str = "",
    new_text: str = "",
) -> str:
    coord_json = json.dumps(coordinates, indent=2, ensure_ascii=False)

    boxes_overlay_js = ""
    if coordinates:
        boxes_js_items = []
        colors = {"name_text": "red", "dob_text": "blue", "name": "red", "dob": "blue"}
        for key, bbox in coordinates.items():
            if isinstance(bbox, list) and len(bbox) == 4:
                color = colors.get(key, "lime")
                boxes_js_items.append(
                    f'{{key:"{key}", ymin:{bbox[0]}, xmin:{bbox[1]}, ymax:{bbox[2]}, xmax:{bbox[3]}, color:"{color}"}}'
                )
        boxes_overlay_js = f"const BOXES = [{','.join(boxes_js_items)}];"

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Debug - Stage: {stage}</title>
<style>
body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }}
h1 {{ color: #e94560; }}
.section {{ background: #16213e; border-radius: 8px; padding: 15px; margin: 15px 0; }}
.section h2 {{ color: #0f3460; background: #e94560; color: white; padding: 8px 12px; border-radius: 4px; display: inline-block; }}
.error {{ background: #4a0e0e; border: 1px solid #e94560; padding: 15px; border-radius: 8px; color: #ff6b6b; }}
.success {{ background: #0e4a1e; border: 1px solid #4caf50; padding: 15px; border-radius: 8px; color: #81c784; }}
pre {{ background: #0a0a1a; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 13px; }}
.image-container {{ position: relative; display: inline-block; margin: 10px; }}
.image-container img {{ max-width: 100%; max-height: 500px; border: 2px solid #333; border-radius: 4px; }}
.overlay-box {{ position: absolute; border: 2px solid; pointer-events: none; }}
.overlay-label {{ position: absolute; top: -20px; left: 0; font-size: 11px; font-weight: bold; padding: 1px 4px; border-radius: 2px; white-space: nowrap; }}
.meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
.meta-item {{ background: #0a0a1a; padding: 8px 12px; border-radius: 4px; }}
.meta-label {{ color: #888; font-size: 12px; }}
.meta-value {{ color: #fff; font-size: 16px; font-weight: bold; }}
.compare {{ display: flex; gap: 20px; flex-wrap: wrap; }}
.compare > div {{ flex: 1; min-width: 300px; }}
</style>
</head>
<body>
<h1>🔍 Debug Report - {stage.upper()}</h1>

<div class="section">
<h2>Thông tin</h2>
<div class="meta">
<div class="meta-item"><div class="meta-label">Stage</div><div class="meta-value">{stage}</div></div>
<div class="meta-item"><div class="meta-label">Render Method</div><div class="meta-value">{render_method or 'N/A'}</div></div>
<div class="meta-item"><div class="meta-label">Processing Time</div><div class="meta-value">{processing_ms or 'N/A'} ms</div></div>
<div class="meta-item"><div class="meta-label">New Text</div><div class="meta-value">{new_text or 'N/A'}</div></div>
</div>
</div>

{"<div class='error'><strong>❌ ERROR:</strong><br>" + error + "</div>" if error else ""}
{"<div class='success'><strong>✅ SUCCESS</strong> - Ảnh đã xử lý thành công.</div>" if stage == "success" else ""}

<div class="section">
<h2>Tọa độ OCR</h2>
<pre>{coord_json}</pre>
</div>

<div class="section">
<h2>Ảnh gốc + Bounding Boxes</h2>
<div class="image-container" id="original-container">
<img id="original-img" src="data:image/jpeg;base64,{image_base64}" onload="drawBoxes()">
</div>
</div>

{f'''<div class="section">
<h2>So sánh Gốc vs Kết quả</h2>
<div class="compare">
<div><h3>Ảnh gốc</h3><img src="data:image/jpeg;base64,{image_base64}" style="max-width:100%;max-height:400px;border:2px solid #333;border-radius:4px;"></div>
<div><h3>Ảnh kết quả ({render_method})</h3><img src="data:image/jpeg;base64,{result_image_base64}" style="max-width:100%;max-height:400px;border:2px solid #4caf50;border-radius:4px;"></div>
</div>
</div>''' if result_image_base64 else ""}

<script>
{boxes_overlay_js}
function drawBoxes() {{
    const img = document.getElementById('original-img');
    const container = document.getElementById('original-container');
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    const dispW = img.clientWidth;
    const dispH = img.clientHeight;
    const scaleX = dispW / w;
    const scaleY = dispH / h;
    
    if (typeof BOXES !== 'undefined') {{
        BOXES.forEach(b => {{
            const maxVal = Math.max(b.ymin, b.xmin, b.ymax, b.xmax);
            let scale = 1000;
            if (maxVal <= 1.0) scale = 1.0;
            else if (maxVal <= 1000) scale = 1000;
            else scale = Math.max(w, h);
            
            const x0 = (b.xmin / scale) * dispW;
            const y0 = (b.ymin / scale) * dispH;
            const x1 = (b.xmax / scale) * dispW;
            const y1 = (b.ymax / scale) * dispH;
            
            const box = document.createElement('div');
            box.className = 'overlay-box';
            box.style.left = x0 + 'px';
            box.style.top = y0 + 'px';
            box.style.width = (x1 - x0) + 'px';
            box.style.height = (y1 - y0) + 'px';
            box.style.borderColor = b.color;
            
            const label = document.createElement('div');
            label.className = 'overlay-label';
            label.textContent = b.key;
            label.style.background = b.color;
            label.style.color = 'white';
            box.appendChild(label);
            
            container.appendChild(box);
        }});
    }}
}}
</script>
</body>
</html>"""
    return html


# ==============================================================================
# 13. HÀM MAIN - KHỞI TẠO VÀ CHẠY BOT
# ==============================================================================
def main() -> None:
    """
    Khởi tạo Application của python-telegram-bot, đăng ký các Handler và
    bắt đầu vòng lặp polling để lắng nghe cập nhật (update) từ Telegram.
    """
    logger.info("Đang khởi tạo Orchestrator Bot...")

    # --- Xây dựng Application từ Token đã cấu hình ---
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # --- Đăng ký handler cho lệnh /start ---
    application.add_handler(CommandHandler("start", start_command))

    # --- Đăng ký handler cho lệnh /key (chỉ admin dùng được) ---
    application.add_handler(CommandHandler("key", key_command))

    # --- Đăng ký handler cho lệnh /listkey (xem + xoá Key, chỉ admin) ---
    application.add_handler(CommandHandler("listkey", listkey_command))

    # --- Đăng ký handler xử lý bấm nút "🗑 Xoá" trong /listkey ---
    application.add_handler(
        CallbackQueryHandler(delete_key_callback, pattern=r"^delkey:")
    )

    # --- Đăng ký handler "Cổng tiếp nhận dữ liệu" từ WebApp ---
    application.add_handler(
        MessageHandler(filters.StatusUpdate.WEB_APP_DATA, web_app_data_handler)
    )

    # --- Đăng ký handler nhận ảnh trực tiếp từ tin nhắn ---
    application.add_handler(
        MessageHandler(filters.PHOTO, photo_message_handler)
    )

    # --- Đăng ký handler bắt lỗi toàn cục ---
    application.add_error_handler(global_error_handler)

    logger.info("Bot đã sẵn sàng. Bắt đầu polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
